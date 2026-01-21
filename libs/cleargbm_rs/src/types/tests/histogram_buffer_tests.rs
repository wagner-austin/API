//! Tests for HistogramBuffer type.

use crate::error::ClearGbmError;
use crate::types::HistogramBuffer;

#[test]
fn test_histogram_buffer_new() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(5_usize);
    assert_eq!(hist.n_bins(), 5_usize);
    for i in 0_usize..5_usize {
        let grad = match hist.gradient_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(grad.abs() < f64::EPSILON);
        let hess = match hist.hessian_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(hess.abs() < f64::EPSILON);
        let count_val = match hist.count(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val, 0_usize);
    }
    Ok(())
}

#[test]
fn test_histogram_buffer_accumulate() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(1_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    let grad = match hist.gradient_sum(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((grad - 0.5_f64).abs() < f64::EPSILON);
    let hess = match hist.hessian_sum(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess - 1.0_f64).abs() < f64::EPSILON);
    let count_val = match hist.count(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count_val, 1_usize);
    Ok(())
}

#[test]
fn test_histogram_buffer_accumulate_multiple() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 0.1_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(0_usize, 0.2_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(0_usize, 0.3_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    let grad = match hist.gradient_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((grad - 0.6_f64).abs() < 1e-10_f64);
    let hess = match hist.hessian_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess - 3.0_f64).abs() < f64::EPSILON);
    let count_val = match hist.count(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count_val, 3_usize);
    Ok(())
}

#[test]
fn test_histogram_buffer_accumulate_out_of_bounds() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    let result = hist.accumulate(5_usize, 0.5_f64, 1.0_f64);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(ClearGbmError::BinIndexOutOfBounds {
            bin: 5_usize,
            n_bins: 3_usize
        })
    ));
    Ok(())
}

#[test]
fn test_histogram_buffer_gradient_sum_out_of_bounds() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = hist.gradient_sum(10_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_hessian_sum_out_of_bounds() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = hist.hessian_sum(10_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_count_out_of_bounds() -> Result<(), ClearGbmError> {
    let hist = HistogramBuffer::new(3_usize);
    let result = hist.count(10_usize);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_histogram_buffer_slices() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 0.1_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(1_usize, 0.2_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(2_usize, 0.3_f64, 3.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    assert_eq!(hist.gradient_sums().len(), 3_usize);
    assert_eq!(hist.hessian_sums().len(), 3_usize);
    assert_eq!(hist.counts().len(), 3_usize);

    assert!((hist.gradient_sums()[0_usize] - 0.1_f64).abs() < f64::EPSILON);
    assert!((hist.hessian_sums()[1_usize] - 2.0_f64).abs() < f64::EPSILON);
    assert_eq!(hist.counts()[2_usize], 1_usize);
    Ok(())
}

#[test]
fn test_histogram_buffer_reset() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match hist.accumulate(1_usize, 0.3_f64, 1.5_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    hist.reset();
    for i in 0_usize..3_usize {
        let grad = match hist.gradient_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(grad.abs() < f64::EPSILON);
        let hess = match hist.hessian_sum(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert!(hess.abs() < f64::EPSILON);
        let count_val = match hist.count(i) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        assert_eq!(count_val, 0_usize);
    }
    Ok(())
}

#[test]
fn test_histogram_buffer_subtract_into() -> Result<(), ClearGbmError> {
    let mut parent = HistogramBuffer::new(3_usize);
    match parent.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(0_usize, 0.3_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match parent.accumulate(1_usize, 0.2_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut child = HistogramBuffer::new(3_usize);
    match child.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut sibling = HistogramBuffer::new(3_usize);
    match sibling.subtract_into(&parent, &child) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // Bin 0: parent (0.8, 2.0, 2), child (0.5, 1.0, 1), sibling should be (0.3, 1.0, 1)
    let grad = match sibling.gradient_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((grad - 0.3_f64).abs() < 1e-10_f64);
    let hess = match sibling.hessian_sum(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert!((hess - 1.0_f64).abs() < f64::EPSILON);
    let count_val = match sibling.count(0_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count_val, 1_usize);

    // Bin 1: parent (0.2, 1.0, 1), child (0, 0, 0), sibling should be (0.2, 1.0, 1)
    let count_val_1 = match sibling.count(1_usize) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    assert_eq!(count_val_1, 1_usize);
    Ok(())
}

#[test]
fn test_histogram_buffer_subtract_into_shape_mismatch() -> Result<(), ClearGbmError> {
    let parent = HistogramBuffer::new(3_usize);
    let child = HistogramBuffer::new(5_usize);
    let mut sibling = HistogramBuffer::new(3_usize);

    let result = sibling.subtract_into(&parent, &child);
    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_histogram_buffer_copy_from() -> Result<(), ClearGbmError> {
    let mut source = HistogramBuffer::new(3_usize);
    match source.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    match source.accumulate(1_usize, 0.3_f64, 2.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    let mut dest = HistogramBuffer::new(3_usize);
    match dest.copy_from(&source) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    assert_eq!(dest.gradient_sums(), source.gradient_sums());
    assert_eq!(dest.hessian_sums(), source.hessian_sums());
    assert_eq!(dest.counts(), source.counts());
    Ok(())
}

#[test]
fn test_histogram_buffer_copy_from_shape_mismatch() -> Result<(), ClearGbmError> {
    let source = HistogramBuffer::new(5_usize);
    let mut dest = HistogramBuffer::new(3_usize);
    let result = dest.copy_from(&source);
    assert!(result.is_err());
    assert!(matches!(result, Err(ClearGbmError::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn test_histogram_buffer_clone() -> Result<(), ClearGbmError> {
    let mut hist = HistogramBuffer::new(3_usize);
    match hist.accumulate(0_usize, 0.5_f64, 1.0_f64) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
    let cloned = hist.clone();
    assert_eq!(hist, cloned);
    Ok(())
}
