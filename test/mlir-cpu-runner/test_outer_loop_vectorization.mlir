// RUN: mlir-opt %s -outer-loop-vectorize -virtual-vec-size 1024 | FileCheck %s

func @matmul6(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
  // CHECK: affine.for {{.*}} = 0 to 512 {
  affine.for %arg3 = 0 to 512 {
    // CHECK: affine.for {{.*}} = 0 to 512 {
    affine.for %arg4 = 0 to 512 {
      affine.for %arg5 = 0 to 512 {
        %a = affine.load %A[%arg3, %arg5] : memref<512x512xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<512x512xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<512x512xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<512x512xf32>
      }
    }
  }
  return
}


func @matmul5(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  affine.for %arg3 = 0 to 2048 {
    // CHECK: affine.for {{.*}} = 0 to 2048 step 1024 {
    affine.for %arg4 = 0 to 2048 {
      // CHECK: affine.for {{.*}} = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32>
        // %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        // %p = mulf %a, %b : f32
        // %co = addf %ci, %p : f32
        // affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}

func @matmul4(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  affine.for %arg3 = 0 to 2048 {
    // CHECK: affine.for {{.*}} = 0 to 2048 step 1024 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, 2049 + %arg4] : memref<2048x2048xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}



func @matmul3(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  // CHECK: affine.for {{.*}} = 0 to 2048 step 1024 {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, 2049 * %arg4] : memref<2048x2048xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}


#map2 = (d0, d1) -> (d1 * 2048 +  d0)
func @matmul2(%A: memref<2048x2048xf32, #map2>, %B: memref<2048x2048xf32, #map2>, %C: memref<2048x2048xf32, #map2>) {
  // CHECK: affine.for {{.*}} = 0 to 2048 step 1024 {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32, #map2>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32, #map2>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32, #map2>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32, #map2>
      }
    }
  }
  return
}


func @matmul(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  affine.for %arg3 = 0 to 2048 {
    // CHECK: affine.for {{.*}} = 0 to 2048 step 1024 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}