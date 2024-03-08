__global__ void matmul(half* v1, half* v2, half* v3) {
  size_t v4 = 32768;
  size_t v5 = 96;
  size_t v6 = 4;
  size_t v7 = 2048;
  size_t v8 = 128;
  size_t v9 = 32;
  size_t v10 = 0;
  size_t v11 = 64;
  size_t v12 = 4224;
  size_t v13 = blockIdx.y;
  size_t v14 = blockIdx.x;
  size_t v15 = v13 * v8;
  size_t v16 = v14 * v8;
  int8_t v17[32768];
  extern __shared__ uint8_t v18[];
  ;
  half* v19 = reinterpret_cast<half*>(v17 + v10);
  half* v20 = reinterpret_cast<half*>(v18 + v10);
  half* v21 = reinterpret_cast<half*>(v18 + v4);
  half* v22 = v20;
  half* v23 = reinterpret_cast<half*>(v22 + 0);
  half* v24 = v1;
  size_t v25 = v15 * v7;
  half* v26 = reinterpret_cast<half*>(v24 + v25);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 128, 32, 2048, 1, 32, 1>(v26, v23).run();
  half* v27 = v21;
  half* v28 = reinterpret_cast<half*>(v27 + 0);
  half* v29 = v2;
  size_t v30 = v16 * v7;
  half* v31 = reinterpret_cast<half*>(v29 + v30);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 32, 128, 1, 2048, 1, 32>(v31, v28).run();
  asm volatile("cp.async.commit_group;" ::);
  half* v32 = reinterpret_cast<half*>(v22 + 4096);
  size_t v33 = v25 + v9;
  half* v34 = reinterpret_cast<half*>(v24 + v33);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 128, 32, 2048, 1, 32, 1>(v34, v32).run();
  half* v35 = reinterpret_cast<half*>(v27 + 4096);
  size_t v36 = v30 + v9;
  half* v37 = reinterpret_cast<half*>(v29 + v36);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 32, 128, 1, 2048, 1, 32>(v37, v35).run();
  asm volatile("cp.async.commit_group;" ::);
  half* v38 = reinterpret_cast<half*>(v22 + 8192);
  size_t v39 = v25 + v11;
  half* v40 = reinterpret_cast<half*>(v24 + v39);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 128, 32, 2048, 1, 32, 1>(v40, v38).run();
  half* v41 = reinterpret_cast<half*>(v27 + 8192);
  size_t v42 = v30 + v11;
  half* v43 = reinterpret_cast<half*>(v29 + v42);
  gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 32, 128, 1, 2048, 1, 32>(v43, v41).run();
  asm volatile("cp.async.commit_group;" ::);
  for (size_t v44 = v10; v44 < v7; v44 += v9) {
    asm volatile("cp.async.wait_group %0;" :: "n"(2));
    __syncthreads();
    size_t v45 = v44 + v5;
    size_t v46 = v45 / v9;
    size_t v47 = v46 % v6;
    size_t v48 = v44 / v9;
    size_t v49 = v48 % v6;
    bool v50 = v45 < v7;
    if (v50) {
      size_t v51 = v46 * v9;
      size_t v52 = v47 * v8;
      size_t v53 = v52 * v9;
      half* v54 = reinterpret_cast<half*>(v22 + v53);
      size_t v55 = v25 + v51;
      half* v56 = reinterpret_cast<half*>(v24 + v55);
      gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 128, 32, 2048, 1, 32, 1>(v56, v54).run();
      half* v57 = reinterpret_cast<half*>(v27 + v53);
      size_t v58 = v51 + v30;
      half* v59 = reinterpret_cast<half*>(v29 + v58);
      gpuLoadSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 32, 128, 1, 2048, 1, 32>(v59, v57).run();
      ;
    };
    asm volatile("cp.async.commit_group;" ::);
    size_t v60 = v49 * v8;
    size_t v61 = v60 * v9;
    half* v62 = reinterpret_cast<half*>(v22 + v61);
    half* v63 = reinterpret_cast<half*>(v27 + v61);
    gpuThreadblockMMA<32, 2, 2, BIR_FP16, 128, 32, 32, 1, BIR_FP16, 32, 128, 1, 32, BIR_FP16, 128, 128, 128, 1>(v62, v63, v19).run();
  }
  half* v64 = reinterpret_cast<half*>(v18 + v10);
  gpuStoreAccumulatorCrosswise_2D<32, 2, 2, BIR_FP16, 128, 128, 128, 1, 128, 1>(v19, v64).run();
  half* v65 = v3;
  size_t v66 = v15 * v12;
  size_t v67 = v66 + v16;
  half* v68 = reinterpret_cast<half*>(v65 + v67);
  gpuStoreSharedMemoryCrosswise_2D<32, 2, 2, BIR_FP16, 128, 128, 128, 1, 4224, 1>(v64, v68).run();
  return;
}



