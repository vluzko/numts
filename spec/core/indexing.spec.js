const indexing = require('../../numts/tensor_core/indexing').indexing;
const numts = require('../../numts/numts');
const random = require('../../numts/random');
const tensor = require('../../numts/tensor').tensor;
const utils = require('../../numts/utils').utils;


describe('Basic calculations.', function () {

  describe('compute_slice_size.', function () {
    it('From failing test.', function () {
      let x = indexing.compute_slice_size([0, 0], [3, 1], [1, 1]);
      expect(x).toBe(3);
    });

    it('Basic checks.', function () {
      expect(indexing.compute_slice_size([0, 0], [2, 2], [1, 1])).toBe(4);
      expect(indexing.compute_slice_size(new Uint32Array([0, 5]), new Uint32Array([6, 10]), new Uint32Array([5, 2]))).toBe(6);
    });
  });

  describe('new_shape_from_slice.', function () {
    it('Basic test.', function () {
      const starts = new Uint32Array(3);
      const ends = new Uint32Array([1, 2, 3]);
      const steps = new Uint32Array(3);
      steps.fill(1);

      const shape = indexing.new_shape_from_slice(starts, ends, steps);
      expect(shape).toEqual(ends);
    });

    it('With steps and offsets.', function () {
      const starts = new Uint32Array([2, 6, 9]);
      const ends = new Uint32Array([10, 12, 11]);
      const steps = new Uint32Array([2, 3, 1]);

      const shape = indexing.new_shape_from_slice(starts, ends, steps);
      const expected = new Uint32Array([4, 2, 2]);
      expect(shape).toEqual(expected);
    });


    it('With unit dimension.', function () {
      const starts = new Uint32Array([2, 6, 10]);
      const ends = new Uint32Array([10, 12, 11]);
      const steps = new Uint32Array([2, 3, 1]);

      const shape = indexing.new_shape_from_slice(starts, ends, steps);
      const expected = new Uint32Array([4, 2, 1]);

      expect(shape).toEqual(expected);
    });

    it('Singleton.', function() {
      const starts = new Uint32Array([0, 0]);
      const ends = new Uint32Array([1, 1]);
      const steps = new Uint32Array([1, 1]);

      const shape = indexing.new_shape_from_slice(starts, ends, steps);
      const expected = new Uint32Array([1, 1]);
      expect(shape).toEqual(expected);
    });
  });

  it('_stride_from_shape.', function () {
    expect(indexing.stride_from_shape([2, 2, 3])).toEqual(new Uint32Array([1, 2, 4]));
    expect(indexing.stride_from_shape([2, 5])).toEqual(new Uint32Array([1, 2]));
  });

  describe('convert_negative_indices.', function () {
    it('basic test.', function () {
      const indices = [-2, [2, 3], [2, -1], [,-3, 4]];
      const shape = [4, 4, 5, 12];
      const expected = [2, [2, 3], [2, 4], [, 9, 4]];
      expect(expected).toEqual(indexing.convert_negative_indices(indices, shape));
    });
  });

  describe('calculate_broadcast_dimensions.', function () {
    it('same dims.', function () {
      for (let i = 0; i < 100; i++) {
        const number_of_dims = random.randint(1, 8).g(0);
        let dims = random.randint(1, 10, [number_of_dims]).data;
        let a = numts.zeros(dims);
        let b = numts.zeros(dims);
        let result = indexing.calculate_broadcast_dimensions(a.shape, b.shape);
        expect(result).toEqual(new Uint32Array(dims), `Input dims: ${dims}. Result: ${result}`);
      }

    });

    it('One shorter.', function () {
      for (let i = 0; i < 10; i++) {
        const number_of_dims = random.randint(1, 8).g(0);
        let dims = random.randint(1, 10, [number_of_dims]).data;
        let a = numts.zeros(dims);
        let b = numts.zeros(dims.slice(number_of_dims - 3));
        let result = indexing.calculate_broadcast_dimensions(a.shape, b.shape);
        expect(result).toEqual(new Uint32Array(dims), `Input dims: ${dims}. Result: ${result}`);
      }
    });

    it('Random ones.', function () {
      for (let i = 0; i < 100; i++) {
        const number_of_dims = random.randint(1, 8).g(0);
        let a_dims = random.randint(1, 10, [number_of_dims]).data;
        let b_dims = a_dims.slice(0);

        for (let j of a_dims) {
          const sw = Math.random() < 1 / a_dims.length;
          if (sw) {
            a_dims[i] = 1;
          }
        }
        const a = numts.zeros(a_dims);
        const b = numts.zeros(b_dims);

        let result = indexing.calculate_broadcast_dimensions(a.shape, b.shape);
        expect(result).toEqual(new Uint32Array(b_dims), `Input dims: ${b_dims}. Result: ${result}`);
      }
    });
  });

  describe('check_indices_are_single_index.', function() {
    it('Simple test.', function() {
      let result = indexing.checks_indices_are_single_index(0, 1, 2);
      expect(result).toBe(true);
    });
  });

  describe('index_to_slice.', function () {
    it('Empty.', function () {
      expect(indexing.index_to_slice([])).toEqual([]);
    });

    it('Single.', function () {
      expect(indexing.index_to_slice(new Uint32Array(1))).toEqual([[0, 1]]);
    });

    it('Simple.', function () {
      expect(indexing.index_to_slice([0, 1])).toEqual([[0, 1], [1, 2]]);
    });

  });

});

describe('index_in_data.', function() {
})

describe('Iterators.', function () {
  const a = numts.arange(30).reshape(5, 6);
  let steps = new Uint32Array(a.shape.length);
  steps.fill(1);
  describe('iorder_index_iterator.', function() {
    it('Basic test.', function() {

      const iter = indexing.iorder_index_iterator(new Uint32Array(a.shape.length), a.shape, steps);
      let prev = -1;
      for (let i of iter) {
        expect(a.g(...i)).toBe(prev + 1);
        prev = a.g(...i);
      }
    });

    it('3 2 5.', function () {
      let iter = indexing.iorder_index_iterator([0, 0], [3, 1], [1, 1]);
      let x = Array.from(iter);
      expect(x).toEqual([[0, 0], [1, 0], [2, 0]]);
    });

    it('one dimensional.', function() {
      const x = Array.from(indexing.iorder_index_iterator([0], [5], [1]));
      expect(x).toEqual([[0], [1], [2], [3], [4]]);
    });

    it('three dimensions.', function() {
      const b = a.reshape(5, 3, 2);
      let steps = new Uint32Array(b.shape.length);
      steps.fill(1);
      const iter = indexing.iorder_index_iterator(new Uint32Array(3), b.shape, steps);
      const x = Array.from(iter);
      expect(x[0]).toEqual(new Uint32Array([0, 0, 0]));
      expect(x[x.length-1]).toEqual(new Uint32Array([4, 2, 1]));
      expect(x[13]).toEqual(new Uint32Array([2, 0, 1]));
    });

    it('Mismatched.', function () {
      const iter = indexing.iorder_index_iterator([0, 5], [4, 5], [5, 5]);
      const arr = Array.from(iter);
      expect(arr.length).toBe(0);
    });

    describe('From failures.', function() {
      it('From QR decomp.', function() {
        const a = numts.from_nested_array([[0], [1], [0.5345224838248488], [0.8017837257372732]]);
        const vals = [...a._iorder_index_iterator()];
        expect(vals).toEqual([new Uint32Array([0, 0]), new Uint32Array([1, 0]), new Uint32Array([2, 0]), new Uint32Array([3, 0])])
      })
    });
  });

  describe('iorder_data_iterator.', function() {
    it('Basic.', function() {
      const iter = indexing.iorder_data_iterator(new Uint32Array(a.shape.length), a.shape, steps, a.stride, 0);
      let prev = -1;
      for (let i of iter) {
        expect(a.data[i]).toBe(prev + 1);
        prev = a.data[i];
      }
    });

    it('Three dimensions.', function() {
      const slice = numts.arange(75).reshape(5, 5, 3).slice([0, 3], [2, 5]);
      const steps = utils.fixed_ones(slice.shape.length);
      const iter = indexing.iorder_data_iterator(new Uint32Array(slice.shape.length), slice.shape, steps, slice.stride, slice.initial_offset);
      const indices = Array.from(iter);
      const data = indices.map(e => slice.data[e]);
      expect(data).toEqual([
        6, 7, 8, 9, 10, 11, 12, 13, 14,
        21, 22, 23, 24, 25, 26, 27, 28, 29,
        36, 37, 38, 39, 40, 41, 42, 43, 44
      ]);
    });
  });

  describe('dorder_data_iterator', function() {
    it('Basic.', function() {
      const slice = a.slice([2, 4]);
      const iter = indexing.dorder_data_iterator(new Uint32Array(slice.shape.length), slice.shape, utils.fixed_ones(slice.shape.length), slice.stride, slice.initial_offset);
      const arr = Array.from(iter);
      expect(arr).toEqual([2, 3, 7, 8, 12, 13, 17, 18, 22, 23, 27, 28]);
    });

    it('Three dimensions.', function() {
      const slice = numts.arange(75).reshape(5, 5, 3).slice([0, 3], [2, 5]);
      const steps = utils.fixed_ones(slice.shape.length);
      const iter = indexing.dorder_data_iterator(new Uint32Array(slice.shape.length), slice.shape, steps, slice.stride, slice.initial_offset);
      const indices = Array.from(iter);
      const data = indices.map(e => slice.data[e]);
      expect(data).toEqual([
        6, 21, 36,
        9, 24, 39,
        12, 27, 42,
        7, 22, 37,
        10, 25, 40,
        13, 28, 43,
        8, 23, 38,
        11, 26, 41,
        14, 29, 44
      ])
    });
  });

  describe('dorder_index_iterator.', function() {
    it('Basic.', function() {
      const iter = indexing.dorder_index_iterator(new Uint32Array(a.shape.length), a.shape, steps);
      const values = Int32Array.from(utils.imap(iter, e => a.g(...e)));
      expect(values).toEqual(a.data);
    });
  });
});

describe('slice.', function () {

  it('basic test.', function () {
    const base_array = numts.arange(16).reshape([4, 4]);
    const s = indexing.slice([[0,2], [1, 3]], base_array.shape, base_array.stride, base_array.offset, base_array.dstride, base_array.initial_offset);
    const slice = base_array.slice([0, 2], [1, 3]);

  });

  it('single value slice.', function () {
    const base_array = numts.arange(16).reshape([4, 4]);
    const slice = base_array.slice(0);
    const expected = numts.arange(4);

    const actual = numts.from_iterable(slice._iorder_value_iterator(), slice.shape, 'int32');
    expect(expected.equals(actual)).toBe(true);
    expect(slice.data).toEqual(base_array.data);
  });

  it('Successive slices.', function () {
    const base_array = numts.arange(16).reshape([4, 4]);
    const first_slice = base_array.slice([0, 2], [1, 3]);
    expect(first_slice.shape).toEqual(new Uint32Array([2, 2]));
    const second_slice = first_slice.slice(0);

    const expected = numts.arange(1, 3);

    const actual = numts.from_iterable(second_slice._iorder_value_iterator(), second_slice.shape, 'int32');
    expect(expected.equals(actual)).toBe(true);
  });

  it('Slice with large steps.', function () {
    const base_array = numts.arange(16).reshape([4, 4]);
    const slice = base_array.slice([0, 4, 2], [1, 3]);

    const expected = numts.from_nested_array([
      [1, 2],
      [9, 10]
    ], 'int32');
    const actual = numts.from_iterable(slice._iorder_value_iterator(), slice.shape, 'int32');

    expect(expected.equals(actual)).toBe(true);

  });

  it('Subdimensions', function () {
    const base_array = numts.arange(120).reshape([4, 5, 6]);
    const first = base_array.slice([0, 4, 2], [1, 3, 2]);
    expect(first.shape).toEqual(new Uint32Array([2, 1, 6]));
    const expected = numts.from_nested_array([
      [[6, 7, 8, 9, 10, 11]],
      [[66, 67, 68, 69, 70, 71]]
    ], 'int32');


    const actual = numts.from_iterable(first._iorder_value_iterator(), first.shape, 'int32');
    expect(expected).toEqual(actual);
  });

  it('Successive slice, large steps.', function () {
    const base_array = numts.arange(120).reshape([4, 5, 6]);
    const first = base_array.slice([0, 4, 2], [1, 3, 2]);
    expect(first.shape).toEqual(new Uint32Array([2, 1, 6]));

    const second = first.slice(null, null, [0, 6, 3]);
    expect(second.shape).toEqual(new Uint32Array([2, 1, 2]));

    const expected = numts.from_nested_array([
      [[6, 9]],
      [[66, 69]]
    ], 'int32');

    const actual = numts.from_iterable(second._iorder_value_iterator(), second.shape, 'int32');
    expect(expected.equals(actual)).toBe(true);
  });

  it('slice with last dropped', function () {
    const a = numts.arange(24).reshape(2, 3, 4);
    const slice = a.slice(...[null, null, 1]);
    expect([...slice._iorder_value_iterator()]).toEqual([1, 5, 9, 13, 17, 21]);
  });

  describe('from failures.', function () {
    it('broadcast_matmul break.', function () {
      const a = numts.arange(24).reshape(2, 3, 4);
      const slice = a.slice(...[1]);
      expect([...slice._iorder_value_iterator()]).toEqual([
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
      ]);
    });

    it('successive slice break.', function(){
      const a = numts.arange(24).reshape(2, 3, 4);
      const first_slice = a.slice(0);
      const second_slice = first_slice.slice(1);
      expect([...second_slice._iorder_value_iterator()]).toEqual([4, 5, 6, 7]);
    });

  });
});

describe('uslice_to_islice.', function() {
  it('empty slice.', function() {
    const shape = [2, 3, 4];
    const [slice, _] = indexing.uslice_to_islice([], shape);
    let i = 0;
    for (let [l, u, s] of slice) {
      expect(l).toBe(0);
      expect(u).toBe(shape[i]);
      expect(s).toBe(1);
      i += 1;
    }
  });

  it('null slice.', function() {
    const shape = [2, 3, 4];
    const [slice, _] = indexing.uslice_to_islice([null], shape);
    let i = 0;
    for (let [l, u, s] of slice) {
      expect(l).toBe(0);
      expect(u).toBe(shape[i]);
      expect(s).toBe(1);
      i += 1;
    }
  });

  it('all types', function() {
    const shape = [5, 5, 5, 5, 5];
    const uslice = [null, 1, [1, 3], [0, 5, 2]];
    const [islice, dims] = indexing.uslice_to_islice(uslice, shape);
    expect(dims).toEqual([1]);
    expect(islice).toEqual([
      [0, 5, 1],
      [1, 2, 1],
      [1, 3, 1],
      [0, 5, 2],
      [0, 5, 1]
    ]);
  });
});

describe('slice_to_bounds.', function() {
  it('Basic.', function() {
    const islice = [[1, 2, 3], [3, 100, 2]];
    const [l, u, s] = indexing.slice_to_bounds(islice);
    expect(l).toEqual(new Uint32Array([1, 3]));
    expect(u).toEqual(new Uint32Array([2, 100]));
    expect(s).toEqual(new Uint32Array([3, 2]));
  });
});
