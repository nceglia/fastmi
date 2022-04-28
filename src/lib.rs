
use pyo3::prelude::*;
use ndarray::*;
use ndarray::Array;

fn zero_inf(value: f32) -> f32 {
    let mut ret = value;
    if value.is_nan() {
        ret = 0.0;
    }
    ret
}

#[pyfunction]
fn mutual_information(a: Vec<usize>, b: Vec<usize>) -> PyResult<f32> {
    let mut max_a = 0;
    for count in &a {
        if count > &max_a {
            max_a = *count;
        }
    }
    let mut max_b = 0;
    for count in &b {
        if count > &max_b {
            max_b = *count;
        }
    }
    let mut joint_distribution = Array2::<f32>::zeros((max_a+1, max_b+1));
    let mut map = HashMap<usize, HashMap<usize, f32>>>
    let it = a.iter().zip(b.iter());
    for a in 1..max_a {
        for b in 1..max_b {
            let mut count = 0.0;
            for (_i,(&x,&y)) in it.enumerate() {
                if x == a && y == b {
                    count += 1.0;
                }
            }
            joint_distribution[[x, y]] += count;
        }
    }
    for (_i,(&x,&y)) in it.enumerate() {
        if x != 0 && y != 0 {
            joint_distribution[[x, y]] += 1.0;
        }
    }
    let normalized_joint_distribution = &joint_distribution / joint_distribution.sum() as f32;
    let jd = normalized_joint_distribution.clone();
    let py = normalized_joint_distribution.sum_axis(Axis(0)).to_vec();
    let px = normalized_joint_distribution.sum_axis(Axis(1)).to_vec();
    let py_i = Array::from_shape_vec((1, py.len()), py).unwrap();
    let px_i = Array::from_shape_vec((px.len(),1),px).unwrap();
    let px_ind = px_i.dot(&py_i);
    let px_ind_pos = normalized_joint_distribution / px_ind;
    let log_px_ind = jd * px_ind_pos.mapv(f32::log2);
    let zerod_log_px_ind = log_px_ind.mapv(zero_inf);
    let mi = zerod_log_px_ind.sum();
    Ok(mi)
}

/// A Python module implemented in Rust.
#[pymodule]
fn fastmi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mutual_information, m)?)?;
    Ok(())
}
