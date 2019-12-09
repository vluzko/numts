const fc = require('fast-check');
const tensor = require('../numts/tensor').tensor;
const indexing = require('../numts/indexing').indexing;

const BROADCAST_ONE_DIM_PROB = 0.1;

function random_shape(max_size, upper) {
  if (max_size === undefined) {
    max_size = 10;
  }

  if (upper === undefined) {
    upper = 10;
  }

  let shape = [];
  for (let i = 0; i < max_size; i++) {
    shape.push(Math.ceil(Math.random() * upper))
  }

  return new Uint32Array(shape);
}

function array_arbitrary(dim_size_min, dim_size_max, min_dims, max_dims) {
  return fc.array(fc.integer(dim_size_min, dim_size_max), min_dims, max_dims).chain(shape => {
    const size = shape.reduce((a, b) => a * b, 1);
    return fc.tuple(fc.array(fc.float(), size, size), fc.constant(shape))
  });
}

function from_shape(shape) {
  const size = indexing.compute_size(shape);
  return fc.tuple(fc.array(fc.float(), size, size), fc.constant(shape));
}

const matrix = array_arbitrary(1, 100, 2, 2);
const small_matrix = array_arbitrary(1, 10, 2, 2);

const thin_matrix = fc.integer(2, 10).chain(n => fc.tuple(fc.constant(n), fc.integer(1, n))).chain(shape => from_shape(shape));

const squarish_array = fc.array(fc.integer(1, 5), 1, 10).chain(shape => {
  const size = shape.reduce((a, b) => a * b, 1);
  return fc.tuple(fc.array(fc.float(), size, size), fc.constant(shape),)
});

const many_dimensions = fc.array(fc.integer(1, 2), 1, 20).chain(shape => {
  const size = shape.reduce((a, b) => a * b, 1);
  return fc.tuple(fc.array(fc.float(), size, size), fc.constant(shape))
});

const large_dimensions = fc.array(fc.integer(100, 10000), 1, 2).chain(shape => {
  const size = shape.reduce((a, b) => a * b, 1);
  return fc.tuple(fc.array(fc.float(), size, size), fc.constant(shape))
});

const broadcastable = small_matrix.chain(
  ([data, shape]) => fc.tuple(fc.constant([data, shape]), fc.integer(1, shape.length))).chain(
  ([[data, shape], dims]) => fc.tuple(fc.constant([data, shape]), fc.array(fc.float(), dims, dims))).chain(
  ([[data, shape], used]) => {
    let shape_2 = new Uint32Array(used.length);
    used.forEach((e, i) => {
      if (e < BROADCAST_ONE_DIM_PROB) {
        shape_2[i] = 1;
      } else {
        shape_2[i] = shape[i];
      }
    });
    return fc.tuple(fc.constant([data, shape]), from_shape(shape));
  });

/**
 * Check a property for two broadcastable arrays.
 * @param {Function} f - The property.
 */
function check_arrays(f) {
  const check = ([[data_1, shape_1], [data_2, shape_2]]) => {
    const a = tensor.from_iterable(data_1, shape_1);
    const b = tensor.from_iterable(data_2, shape_2);
    return f(a, b);
  };

  const params = {
    numRuns: 15
  };

  fc.assert(fc.property(broadcastable, check), params);
}

function check_random_array(f) {
  const check = ([data, shape]) => {
    const a = tensor.from_iterable(data, shape);
    return f(a);
  }
  const params = {
    numRuns: 15
  };
  fc.assert(fc.property(squarish_array, check), params);  
  fc.assert(fc.property(many_dimensions, check), params);
  fc.assert(fc.property(large_dimensions, check), params);
}

/**
 * Perform a property-based check on matrices specifically (meaning 2-dimensional tensors).
 * @param {*} f             - A boolean function computing the desired properties.
 * @param {string} filter   - A string filter on the types of matrices to check.
 */
function check_matrix(f, filter = '') {
  const params = {
    numRuns: 15
  };

  const check = ([data, shape]) => {
    const a = tensor.from_iterable(data, shape);
    return f(a);
  }

  if (filter === 'only_small') {
    fc.assert(fc.property(small_matrix, check), params);
  } else if (filter === 'thin') {
    fc.assert(fc.property(thin_matrix, check), params);
  } else {
    fc.assert(fc.property(small_matrix, check), params);
    fc.assert(fc.property(thin_matrix, check), params);
    fc.assert(fc.property(matrix, check), params);
  }
}

module.exports = {check_arrays, random_shape, check_random_array, check_matrix};