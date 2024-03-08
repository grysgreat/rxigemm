// wmma + fake pipeline

// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 3.58903ms
// TFLOPS: 32.9838

// 3090
// Setting to 4 stages.
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 5.69767ms
// TFLOPS: 20.7769

//nvcc -arch=sm_70 -std=c++17 -Xcompiler -fopenmp matmul-int8.cu main.cu -o test -lcublas && ./test stages 4

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{

    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = A[(by * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemC(float *smem, half *C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
    }
}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void storeAccum(float *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> *frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }

}

constexpr bool GEMM_OP_T = true;
constexpr bool GEMM_OP_N = false;
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_SIZE_M, int WARP_SIZE_N, int STAGE, bool NoTransA, bool NoTransB, bool RowMajorC>
__global__ void GEMMI8TCU(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K)
{

  constexpr int WARP_SIZE = 32;
  constexpr int TC_SIZE = 16;
  constexpr int CP_SIZE_BYTES = 16;
  constexpr int WAPR_NUM_N = BLOCK_SIZE_N / WARP_SIZE_N;
  constexpr int WAPR_NUM_M = BLOCK_SIZE_M / WARP_SIZE_M;
  constexpr int WAPR_NUM     = WAPR_NUM_M * WAPR_NUM_N;

  static_assert(NoTransA == GEMM_OP_T, "NoTransA == GEMM_OP_T");
  static_assert(NoTransB == GEMM_OP_N, "NoTransB == GEMM_OP_N");
  static_assert(RowMajorC == GEMM_OP_T, "RowMajorC == GEMM_OP_T");

  int warp_id = threadIdx.x/WARP_SIZE;
  int lane_id = threadIdx.x%WARP_SIZE;

  __shared__ int8_t SLB[STAGE * (BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N)];

  int8_t* smem_a[2];
  int8_t* smem_b[2];

  smem_a[0] = SLB;
  smem_a[1] = SLB + BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[0] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M;
  smem_b[1] = SLB + STAGE*BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N;

  const int BCM = BLOCK_SIZE_M * blockIdx.y;
  const int BCN = BLOCK_SIZE_N * blockIdx.x;

  const int LDA = NoTransA ? K : M;
  const int LDB = NoTransB ? N : K;
  const int LDC = RowMajorC ? N : M;

  const int WCM = warp_id / WAPR_NUM_N;
  const int WCN = warp_id % WAPR_NUM_N;

  const int BLOCK_K_LOOP = K / BLOCK_SIZE_K;

  const int8_t* BA = A + BCM * LDA;
  const int8_t* BB = B + BCN * LDB;
  int8_t* BC = C + BCM * LDC + BCN;
  int8_t* BWC = BC + WCM * WARP_SIZE_M * LDC + WCN * WARP_SIZE_N;

  constexpr int WARP_M_LOOP = WARP_SIZE_M / TC_SIZE;
  constexpr int WARP_N_LOOP = WARP_SIZE_N / TC_SIZE;
  constexpr int WARP_K_LOOP = BLOCK_SIZE_K / TC_SIZE;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::row_major> frag_a[WARP_M_LOOP][WARP_K_LOOP];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::col_major> frag_b[WARP_K_LOOP][WARP_N_LOOP];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int> frag_c[WARP_M_LOOP][WARP_N_LOOP];

  #pragma unroll
  for (int i = 0; i < WARP_M_LOOP; i++) {
      #pragma unroll
      for (int j = 0; j < WARP_N_LOOP; j++) {
          nvcuda::wmma::fill_fragment(frag_c[i][j], 0);
      }
  }  

  constexpr int WARP_SIZE_X = 2;
  int lane_id_x = lane_id % (WARP_SIZE_X); // [0,2]
  int lane_id_y = lane_id / (WARP_SIZE_X); // [0,16]

  const int8_t* load_gmem_addr_a, *load_gmem_addr_b;
  int store_smem_addr_a, store_smem_addr_b;
  int k;

  k = 0;

  #pragma unroll
  for(int j = 0; j < BLOCK_SIZE_K/(CP_SIZE_BYTES*WARP_SIZE_X); j++){
    #pragma unroll
    for(int i=warp_id; i<(BLOCK_SIZE_M/TC_SIZE); i+=WAPR_NUM)
    {
      load_gmem_addr_a = BA + (i*TC_SIZE + lane_id_y) * LDA + k*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
      store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (i*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
      asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTES));
    }
    
    #pragma unroll
    for(int i=warp_id; i<(BLOCK_SIZE_N/TC_SIZE); i+=WAPR_NUM)
    {
      load_gmem_addr_b = BB + (i*TC_SIZE + lane_id_y) * LDB + k*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
      store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (i*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
      asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTES));
    }
  }

  #pragma unroll
  for(k=1; k<BLOCK_K_LOOP; k++){
    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < BLOCK_SIZE_K/(CP_SIZE_BYTES*WARP_SIZE_X); j++){
      #pragma unroll
      for(int i=warp_id; i<(BLOCK_SIZE_M/TC_SIZE); i+=WAPR_NUM)
      {
        load_gmem_addr_a = BA + (i*TC_SIZE + lane_id_y) * LDA + k*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
        store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (i*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTES));
      }
      
      #pragma unroll
      for(int i=warp_id; i<(BLOCK_SIZE_N/TC_SIZE); i+=WAPR_NUM)
      {
        load_gmem_addr_b = BB + (i*TC_SIZE + lane_id_y) * LDB + k*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
        store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (i*TC_SIZE + lane_id_y)*BLOCK_SIZE_K + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTES));
      }
    }

    #pragma unroll
    for(int yi=0; yi<WARP_M_LOOP; yi++){
      #pragma unroll
        for(int ki=0; ki<WARP_K_LOOP; ki++){
          nvcuda::wmma::load_matrix_sync(frag_a[yi][ki], &smem_a[(k-1)%2][(WCM*WARP_SIZE_M+yi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
        }
    }

    #pragma unroll
    for(int ki=0; ki<WARP_K_LOOP; ki++){
      #pragma unroll
        for(int xi=0; xi<WARP_N_LOOP; xi++){
          nvcuda::wmma::load_matrix_sync(frag_b[ki][xi], &smem_b[(k-1)%2][(WCN*WARP_SIZE_N+xi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
        }
    }

    #pragma unroll
    for(int ki=0; ki<WARP_K_LOOP; ki++)
      #pragma unroll
      for(int yi=0; yi<WARP_M_LOOP; yi++){
        #pragma unroll
        for(int xi=0; xi<WARP_N_LOOP; xi++){
          nvcuda::wmma::mma_sync(frag_c[yi][xi], frag_a[yi][ki], frag_b[ki][xi], frag_c[yi][xi]);
        }
      }
  }

  asm ("cp.async.commit_group;\n" ::);
  asm ("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  k = BLOCK_K_LOOP -1;
  #pragma unroll
  for(int yi=0; yi<WARP_M_LOOP; yi++){
    #pragma unroll
      for(int ki=0; ki<WARP_K_LOOP; ki++){
        nvcuda::wmma::load_matrix_sync(frag_a[yi][ki], &smem_a[(k)%2][(WCM*WARP_SIZE_M+yi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
      }
  }

  #pragma unroll
  for(int ki=0; ki<WARP_K_LOOP; ki++){
    #pragma unroll
      for(int xi=0; xi<WARP_N_LOOP; xi++){
        nvcuda::wmma::load_matrix_sync(frag_b[ki][xi], &smem_b[(k)%2][(WCN*WARP_SIZE_N+xi*TC_SIZE)*BLOCK_SIZE_K+ki*TC_SIZE], BLOCK_SIZE_K);
      }
  }

  #pragma unroll
  for(int ki=0; ki<WARP_K_LOOP; ki++)
    #pragma unroll
    for(int yi=0; yi<WARP_M_LOOP; yi++){
      #pragma unroll
      for(int xi=0; xi<WARP_N_LOOP; xi++){
        nvcuda::wmma::mma_sync(frag_c[yi][xi], frag_a[yi][ki], frag_b[ki][xi], frag_c[yi][xi]);
      }
    }

  int gmem_lane_id_x = lane_id % 4; // [0,4]
  int gmem_lane_id_y = lane_id / 4; // [0 8]
  #pragma unroll
  for(int yi=0; yi<WARP_M_LOOP; yi++)
    #pragma unroll
    for(int xi=0; xi<WARP_N_LOOP; xi++)
    {
      for(int tc_yi=0; tc_yi<2; tc_yi++){
        for(int tc_xi=0; tc_xi<2; tc_xi++){
          auto* store_gmem_addr = reinterpret_cast<char2*>(BWC + (yi*TC_SIZE + tc_yi*TC_SIZE/2 + gmem_lane_id_y) * LDC + xi*TC_SIZE + tc_xi*TC_SIZE/2 + gmem_lane_id_x*2);
          char2 tmp_char2;
          tmp_char2.x = static_cast<int8_t>(frag_c[yi][xi].x[tc_xi*4+tc_yi*2+0]); 
          tmp_char2.y = static_cast<int8_t>(frag_c[yi][xi].x[tc_xi*4+tc_yi*2+1]);
          *store_gmem_addr = tmp_char2; 
        }
      }
    }
}            

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ half shared_storage[];
    half *SA1 = reinterpret_cast<half *>(shared_storage);
    half *SA2 = SA1 + MI * KI;
    half *SA3 = SA2 + MI * KI;
    half *SA4 = SA3 + MI * KI;
    half *SB1 = SA4 + MI * KI;
    half *SB2 = SB1 + NI * KI;
    half *SB3 = SB2 + NI * KI;
    half *SB4 = SB3 + NI * KI;

    half s;
    float *SC = reinterpret_cast<float *>(shared_storage);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }

    // prologue
    loadSmemA(SA1, A, M, K, 0);
    loadSmemB(SB1, B, N, K, 0);

    loadSmemA(SA2, A, M, K, 1);
    loadSmemB(SB2, B, N, K, 1);

    loadSmemA(SA3, A, M, K, 2);
    loadSmemB(SB3, B, N, K, 2);

    for (int ko = 0; ko < K / KI; ko += 4)
    {
        __syncthreads();
        if (ko + 3 < K / KI)
        {
            loadSmemA(SA4, A, M, K, ko + 3);
            loadSmemB(SB4, B, N, K, ko + 3);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA1, ki);
            loadFragB(FragB, SB1, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }

        __syncthreads();
        if (ko + 4 < K / KI)
        {
            loadSmemA(SA1, A, M, K, ko + 4);
            loadSmemB(SB1, B, N, K, ko + 4);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA2, ki);
            loadFragB(FragB, SB2, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }

        __syncthreads();
        if (ko + 5 < K / KI)
        {
            loadSmemA(SA2, A, M, K, ko + 5);
            loadSmemB(SB2, B, N, K, ko + 5);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA3, ki);
            loadFragB(FragB, SB3, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }

        __syncthreads();
        if (ko + 6 < K / KI)
        {
            loadSmemA(SA3, A, M, K, ko + 6);
            loadSmemB(SB3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA4, ki);
            loadFragB(FragB, SB4, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}