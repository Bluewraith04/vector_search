#[cfg(target_arch="x86_64")]
use std::arch::x86_64::*;


#[inline(always)] 
pub fn compute_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { squared_euclidean_avx2(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        squared_euclidean_distance(a, b)
    }
}

// --- Unoptimized Euclidean Distance --- 
fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x-y).powi(2)).sum()
}

// --- AVX2 Optimized Euclidean Distance ---
#[cfg(target_arch = "x86_64")]
unsafe fn squared_euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_v = _mm256_setzero_ps();

    let n = a.len();

    let mut i = 0;
    while i + 8 <= n {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        let va = _mm256_loadu_ps(a_ptr);
        let vb = _mm256_loadu_ps(b_ptr);

        let diff = _mm256_sub_ps(va, vb);

        let sq = _mm256_mul_ps(diff, diff);
        
        sum_v = _mm256_add_ps(sum_v, sq);

        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
    let mut total_sum: f32 = temp.iter().sum();

    while i < n {
        let diff = a[i] - b[i];
        total_sum += diff * diff;
        i += 1;
    }

    total_sum
}
