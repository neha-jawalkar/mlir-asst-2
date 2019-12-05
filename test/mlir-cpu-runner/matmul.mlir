#map = (d0, d1) -> (4096 * d1 + d0)
func @matmul(%A: memref<4096x4096xf32, #map>, %B: memref<4096x4096xf32, #map>, %C: memref<4096x4096xf32, #map>) {
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 4096 {
      affine.for %arg5 = 0 to 4096 {
        %a = affine.load %A[%arg3, %arg5] : memref<4096x4096xf32, #map>
        %b = affine.load %B[%arg5, %arg4] : memref<4096x4096xf32, #map>
        %ci = affine.load %C[%arg3, %arg4] : memref<4096x4096xf32, #map>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<4096x4096xf32, #map>
      }
    }
  }
  return
}

func @main() -> f32 {
  %A = alloc() : memref<4096x4096xf32, #map>
  %B = alloc() : memref<4096x4096xf32, #map>
  %C = alloc() : memref<4096x4096xf32, #map>
  %cf1 = constant 1.00000e+00 : f32
  %cf0 = constant 0.0 : f32
  linalg.fill(%A, %cf1) : memref<4096x4096xf32, #map>, f32
  linalg.fill(%B, %cf1) : memref<4096x4096xf32, #map>, f32
  linalg.fill(%C, %cf1) : memref<4096x4096xf32, #map>, f32
  call @matmul(%A, %B, %C) : (memref<4096x4096xf32, #map>, memref<4096x4096xf32, #map>, memref<4096x4096xf32, #map>) -> ()
  // call @print_memref_2d_f32(%C): (memref<4096x4096xf32, #map>) -> ()
  return %cf1 : f32
}

func @print_memref_2d_f32(memref<4096x4096xf32, #map>)
func @print_memref_0d_f32(memref<f32>)