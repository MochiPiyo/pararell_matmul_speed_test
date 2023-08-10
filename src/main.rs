

#[derive(Debug, PartialEq)]
struct Matrix<const N: usize, const M: usize> {
    body: Box<[[f32; M]; N]>
}


impl<const N: usize, const M: usize> Matrix<N, M> {
    // Function to create a new matrix
    fn new(matrix: [[f32; M]; N]) -> Self {
        Self { body: Box::new(matrix) }
    }

    // Function to create a new random matrix
    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        // this causes stack overflow. because Box alocate stack and move it to heap.
        //let mut random_matrix: Box<[[f32; M]; N]> = Box::new([[0.0; M]; N]); // 初期化された二次元配列
        
        // ヒープに直接確保する。
        let mut ramdom_matrix: Box<[[f32; M]; N]> = new_direct_matrix_box();
        for i in ramdom_matrix.iter_mut() {
            for j in i.iter_mut() {
                *j = rng.gen_range(0.0 .. 1.0);
            }
        }
        Self {
            body: ramdom_matrix
        }
    }

    fn transpose(&self) -> Matrix<M, N> {
        let mut result = Matrix::<M, N> {
            body: Box::new([[0.0; N]; M]),
        };

        for i in 0..N {
            for j in 0..M {
                result.body[j][i] = self.body[i][j];
            }
        }

        result
    }

    // Display the matrix
    fn display(&self) {
        for row in &*self.body {
            for val in row {
                print!("{} ", val);
            }
            println!();
        }
    }
}

fn new_direct_matrix_box<const N: usize, const M: usize>() -> Box<[[f32; M]; N]> {
    unsafe {
        let ptr = std::alloc::alloc(std::alloc::Layout::new::<[[f32; M]; N]>()) as *mut [[f32; M]; N];
        Box::from_raw(ptr)
    }
}
fn new_direct_matrix_box_fill<const N: usize, const M: usize>(init: f32) -> Box<[[f32; M]; N]> {
    let mut a = new_direct_matrix_box();
    for i in a.iter_mut() {
        for j in i.iter_mut() {
            *j = 0.0;
        }
    }
    a
}

fn matmul_normal<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0)
    };
    for i in 0..N {
        for j in 0..P {
            for k in 0..M {
                result.body[i][j] += a.body[i][k] * b.body[k][j];
            }
        }
    }
    result
}

fn matmul_colum_style_parameter_parallel<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    parameter: &Matrix<P, M>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0)
    };
    for i in 0..N {
        result.body[i].par_iter_mut().enumerate().for_each(|(j, column)| {
            for k in 0..M {
                *column += a.body[i][k] * parameter.body[j][k];
            }
        })
    }
    result
}


fn matmul_ikj<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0),
    };
    for i in 0..N {
        for k in 0..M {
            for j in 0..P {
                result.body[i][j] += a.body[i][k] * b.body[k][j];
            }
        }
    }
    result
}
#[test]
fn matmul_ijk_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, matmul_ikj(&a, &b))
}

use rand::Rng;
use rayon::prelude::*;


//実質Batch並列
fn matmul_data_parallel<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0),
    };

    result.body.par_iter_mut().enumerate().for_each(|(i, row)| {
        for k in 0..M {
            for j in 0..P {
                row[j] += a.body[i][k] * b.body[k][j];
            }
        }
    });
    result
}
#[test]
fn matmul_data_parallel_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, matmul_data_parallel(&a, &b))
}

fn matmul_vector_component_parallel<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0),
    };
    for row in 0..N {
        //横一列に並列化
        result.body[row].par_iter_mut().enumerate().for_each(|(j, result_elem)| {
            for k in 0..M {
                *result_elem += a.body[row][k] * b.body[k][j];
            }
        });
    }
    result
}

#[test]
fn matmul_vector_component_parallel_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, matmul_vector_component_parallel(&a, &b))
}

use std::arch::x86_64::*;

// test fail!
fn matmul_simd<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0),
    };

    for i in 0..N {
        for j in 0..P {
            let mut sum = unsafe { _mm256_setzero_ps() };
            for k in (0..M).step_by(8) {
                let a_vec = unsafe { _mm256_loadu_ps(&a.body[i][k]) };
                let b_vec = unsafe { _mm256_broadcast_ss(&b.body[k][j]) };
                sum = unsafe { _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec)) };
            }
            let mut temp = [0.0; 8];
            unsafe { _mm256_storeu_ps(temp.as_mut_ptr(), sum) };
            result.body[i][j] = temp.iter().sum();
        }
    }

    result
}
#[test]
fn matmul_simd_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, matmul_simd(&a, &b))
}

// test fail!
fn matmul_ikj_simd_parallel<const N: usize, const M: usize, const P: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<M, P>,
) -> Matrix<N, P> {
    let mut result = Matrix::<N, P> {
        body: new_direct_matrix_box_fill(0.0),
    };

    const CHUNK_SIZE: usize = 8;// f32 * 8 = 256bit
    result.body.par_iter_mut().enumerate().for_each(|(i, row)| {
        for k in 0..M {
            // Load a.body[i][k] into a SIMD register
            let a_val = unsafe { _mm256_set1_ps(a.body[i][k]) };

            for j in (0..P).step_by(CHUNK_SIZE) {
                // Load b.body[k][j..j+CHUNK_SIZE] into a SIMD register
                let b_vals = unsafe { _mm256_loadu_ps(&b.body[k][j]) };

                // Multiply a_val with b_vals
                let mul_result = unsafe { _mm256_mul_ps(a_val, b_vals) };

                // Accumulate the multiplication results in result.body[i][j..j+CHUNK_SIZE]
                let existing_results = unsafe { _mm256_loadu_ps(&row[j]) };
                let updated_results = unsafe { _mm256_add_ps(existing_results, mul_result) };
                unsafe {
                    _mm256_storeu_ps(&mut row[j], updated_results);
                }
            }
        }
    });

    result
}
#[test]
fn matmul_ikj_simd_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, matmul_ikj_simd_parallel(&a, &b))
}



fn both_column_jki_parallel<const N: usize, const M: usize, const P: usize>(
    a_transposed: &Matrix<M, N>,
    b_transposed: &Matrix<P, M>
) -> Matrix<P, N> {
    let mut result = Matrix::<P, N> {
        body: new_direct_matrix_box_fill(0.0),
    };

    result.body.par_iter_mut().enumerate().for_each(|(j, row)| {
        for k in 0..M {
            for i in 0..N {
                row[i] += a_transposed.body[k][i] * b_transposed.body[j][k];
            }
        }
    });
    result
}
#[test]
fn both_column_jki_parallel_test() {
    let a = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let b = Matrix::new(
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    );
    let c = Matrix::new(
        [
            [7.0, 10.0],
            [15.0, 22.0]
        ]
    );
    assert_eq!(c, both_column_jki_parallel(&a, &b))
}




fn main() {
    
    const N: usize = 1;
    const M: usize = 100000;
    const P: usize = 30000;
    println!("N = {}, M = {}, P = {}", N, M, P);

    let start = time::Instant::now();
    let a: Matrix<N, M> = Matrix::new_random();
    println!("build ramdom matrix: {}", start.elapsed());
    let b: Matrix<M, P> = Matrix::new_random();
    // matmul of Matrix<100, 100> is O(1,000,000) because matmul is O(N^3)


    /*
    let start = time::Instant::now();
    let c_normal = matmul_normal(&a, &b);
    println!("{}", c.body[0][0]);
    println!("normal: {}", start.elapsed());

    let start = time::Instant::now();
    let c = matmul_ikj(&a, &b);
    println!("{}", c.body[0][0]);
    println!("ikj: {}", start.elapsed());
    */
    let b_transposed = b.transpose();
    let a_transposed = a.transpose();


    let start = time::Instant::now();
    let c = matmul_colum_style_parameter_parallel(&a, &b_transposed);
    println!("column parallel: {}", start.elapsed());

    let start = time::Instant::now();
    let d = matmul_data_parallel(&a, &b);
    println!("data parallel: {}", start.elapsed());

    assert_eq!(c, d);

    
    let start = time::Instant::now();
    let e = both_column_jki_parallel(&a_transposed, &b_transposed);
    println!("both column style jki: {}", start.elapsed());
    assert_eq!(c, e.transpose());

    /*
    let start = time::Instant::now();
    let c = matmul_vector_component_parallel(&a, &b);
    println!("{}", c.body[0][0]);
    println!("vector component wise parallel: {}", start.elapsed());

    
    let start = time::Instant::now();
    let c = matmul_simd(&a, &b);
    println!("{}", c.body[0][0]);
    println!("simd: {}", start.elapsed());
    

    let start = time::Instant::now();
    let c = matmul_ikj_simd_parallel(&a, &b);
    println!("{}", c.body[0][0]);
    println!("simd ikj: {}", start.elapsed());
    */
}


