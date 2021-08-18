const numts = require('../numts/numts');
const errors = require('../numts/tensor').errors;
const tensor = numts.tensor;
const binary_ops = require('../numts/tensor_core/binary_ops');
const utils = require('../numts/utils').utils;

describe('Constructors and factories.', function () {
  describe('Shapes.', function () {

    class Sample {
      constructor(input_array, input_shape, expected) {
        this.input_array = input_array;
        this.input_shape = input_shape;
        this.expected = expected;
      }
    }

    // TODO: Tests for wrong input types.
    const passing_samples = new Map([
      ['null', new Sample([], null, {
        'shape': new Uint32Array([0]),
        'length': 0,
      })],
      ['flat', new Sample([1, 2, 3, 4], [4], {
        'shape': new Uint32Array([4]),
        'length': 4,
      })],
      ['two-dimensional', new Sample([1, 2, 3, 4, 5, 6], [2, 3], {
        'shape': new Uint32Array([2, 3]),
        'length': 6,
      })],
    ]);

    passing_samples.forEach((sample, n) => {
      it(`Testing pass sample ${n}.`, function () {
        const array = numts.array(sample.input_array, sample.input_shape);
        for (let prop in sample.expected) {
          expect(array[prop]).toEqual(sample.expected[prop]);
        }
      });
    });

    describe('Failing tests.', function () {
      it('Data not an array.', function () {
        expect(() => numts.array(1)).toThrow(new errors.BadData());
      });

      it('Data not numeric.', function () {
        expect(() => numts.array(['asd'])).toThrow(new errors.BadData());
      });

      it('Wrong shape type.', function () {
        expect(() => numts.array([], 'asdf')).toThrow(new Error('Shape must be an int, an array of numbers, or a Uint32Array.'));
      });

      it('No shape parameter.', function () {
        const array = numts.array([1, 2, 3, 4]);
        expect(array.shape).toEqual(new Uint32Array([4]));
        expect(array.length).toBe(4);
      });

      describe('Wrong dimensions.', function () {
        it('Wrong length.', function () {
          expect(() => numts.array([1, 2, 3], [2, 2])).toThrow(new errors.MismatchedShapeSize());
        });

        it('Null shape', function () {
          expect(() => numts.array([1, 2, 3], [null])).toThrow(new Error('Shape array must be numeric.'));
        });

        it('Empty shape.', function () {
          expect(() => numts.array([1, 2, 3], [])).toThrow(new errors.MismatchedShapeSize());
        });
      });
    });
  });

  describe('zeros.', function() {

  });

  describe('eye.', function() {
    it('Two-dimensional', function() {
      const a = numts.eye(10);
      for (let [i, j] of a._iorder_index_iterator()) {
        if (i === j) {
          expect(a.g(i, j)).toBe(1);
        } else {
          expect(a.g(i, j)).toBe(0);
        }
      }
    });
  });

  describe('filled.', function() {
    it('Column vector.', function() {
      const a = numts.arange(4).reshape(4, 1);
      expect(a.stride).toEqual(new Uint32Array([1, 4]));
      const arr = [...a._iorder_data_iterator()];
      expect(arr).toEqual([0, 1, 2, 3]);
    });
  });

  describe("from_nested_array.", function () {

    it("hand array.", function () {
      let nested = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]];
      let tensor = numts.from_nested_array(nested);
      expect(tensor.shape).toEqual(new Uint32Array([2, 2, 2]));
      expect(tensor.g(0, 0, 0)).toBe(0);
      expect(tensor.g(0, 0, 1)).toBe(1);
      expect(tensor.g(0, 1, 0)).toBe(2);
      expect(tensor.g(0, 1, 1)).toBe(3);
      expect(tensor.g(1, 0, 0)).toBe(4);
      expect(tensor.g(1, 0, 1)).toBe(5);
      expect(tensor.g(1, 1, 0)).toBe(6);
      expect(tensor.g(1, 1, 1)).toBe(7);

    });

    it("larger array", function () {
      // Create nested array with dimensions 3 x 2 x 3 x 5
      let nested = [1, 2, 3].map(
        x => [x, x+1].map(
          y => [y, y+1, y+2, y+3].map(
            z => [y + 2, y+3, y+4, y+5, y+6]
          ),
        ),
      );

      let good_nested = numts.from_nested_array(nested);
      expect(good_nested.shape).toEqual(new Uint32Array([3, 2, 4, 5]));
      for (let indices of good_nested._iorder_index_iterator()) {
        let expected = utils._nested_array_value_from_index(nested, indices);
        let actual = good_nested.g(...indices);
        expect(actual).toBe(expected, `index: ${indices}`);

        expected = nested[indices[0]][indices[1]][indices[2]][indices[3]];
        actual = good_nested.g(...indices);
        expect(actual).toBe(expected, `index: ${indices}`);
      }
      expect(nested[0][0][0][0]).toBe(good_nested.g(0, 0, 0, 0));
      expect(nested[2][0][0][0]).toBe(good_nested.g(2, 0, 0, 0));
    });
  });

  describe('to_nested_array.', function() {
    it('Basic.', function() {
      let x = numts.arange(10).reshape(2, 5);
      let y = x.to_nested_array();
      expect(y).toEqual([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
      ]);
    });
  });

  describe('to_json.', function() {

  });

  describe('from_json', function() {

  });

});

describe('Indices.', function () {
  it('_compute_real_index.', function () {
    expect(numts.zeros([2, 2])._compute_real_index([1, 1])).toBe(3);
    expect(numts.zeros([2, 3, 4, 5]));
  });

  describe('g.', function () {
    it('3-dims.', function () {
      const array = numts.arange(27).reshape([3, 3, 3]);
      expect(array.g(1, 1, 1)).toBe(13);
      expect(array.g(1, 0, 1)).toBe(10);
      expect(array.g(2, 1, 0)).toBe(21);
    });

    it('4-dims.', function () {
      const array = numts.arange(120).reshape([2, 3, 4, 5]);
      expect(array.g(1, 2, 3, 4)).toBe(119);
      expect(array.g(0, 2, 3, 4)).toBe(59);
      expect(array.g(0, 0, 0, 0)).toBe(0);
    });

    it('negatives indices.', function () {
      const array = numts.arange(40).reshape([5, 4, 2]);
      expect(array.g(-1, -2, -1)).toBe(37);
    });

    describe('From failures.', function() {
      it('From QR decomp.', function() {
        const a = numts.from_nested_array([[0, 1, 0.5345224838248488, 0.8017837257372732]])
        const index = new Uint32Array([0, 1]);
        const b = a.g(...index);
        expect(b).toBe(1);
      });
    });
  });

});

describe('Iterators.', function () {

  describe('data_index_iterator.', function () {

    it('one-dimensional.', function () {
      let tensor = numts.arange(0, 10);
      let indices = [...tensor._iorder_index_iterator()];
      let real_indices = [...tensor._iorder_data_iterator()];

      indices.forEach((e, i) => {
        let real_index = real_indices[i];
        let computed = tensor._compute_real_index(e);
        expect(real_index).toBe(computed);
      });
    });

    it('two-dimensional.', function () {
      let tensor = numts.arange(0, 10).reshape([5, 2]);
      let indices = [...tensor._iorder_index_iterator()];
      let real_indices = [...tensor._iorder_data_iterator()];

      indices.forEach((e, i) => {
        let real_index = real_indices[i];
        let computed = tensor._compute_real_index(e);
        expect(real_index).toBe(computed);
      });
    });

    it('four-dimensional.', function () {
      let tensor = numts.arange(0, 16).reshape(new Uint32Array([2, 2, 2, 2]));
      let indices = [...tensor._iorder_index_iterator()];
      let real_indices = [...tensor._iorder_data_iterator()];

      indices.forEach((e, i) => {
        let real_index = real_indices[i];
        let computed = tensor._compute_real_index(e);
        expect(real_index).toBe(computed);
      });

    });
  });

  describe('_index_iterator.', function () {
    it('two-dimensional', function () {
      let array = numts.zeros([2, 2]);
      const expected = [[0, 0], [0, 1], [1, 0], [1, 1]].map(e => new Uint32Array(e));
      const actual = Array.from(array._iorder_index_iterator());
      expect(actual).toEqual(expected);
    });
  });

  it('_iorder_value_iterator.', function () {
    let array = numts.array([1, 2, 3, 4]);
    const expected = [1, 2, 3, 4];
    const actual = Array.from(array._iorder_value_iterator());
    expect(actual).toEqual(expected);

  });

});

describe('Slicing.', function () {

  describe('slice.', function () {
    it('empty slice.', function () {
      let a = numts.arange(15).reshape(3, 5);
      let b = a.slice();
      expect(a.equals(b));
    });

    it('basic test.', function () {
      const base_array = numts.arange(16).reshape([4, 4]);
      const slice = base_array.slice([0, 2], [1, 3]);

      const expected = numts.from_nested_array([
        [1, 2],
        [5, 6]
      ], 'int32');

      const actual = numts.from_iterable(slice._iorder_value_iterator(), slice.shape, 'int32');
      expect(expected.equals(actual)).toBe(true);
      expect(slice.data).toEqual(base_array.data);
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

    describe('From failures.', function () {
      it('1: broadcast_matmul break.', function () {
        const a = numts.arange(24).reshape(2, 3, 4);
        const slice = a.slice(...[1]);
        expect([...slice._iorder_value_iterator()]).toEqual([
          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ]);
      });

      it('2: successive slice break.', function(){
        const a = numts.arange(24).reshape(2, 3, 4);
        const first_slice = a.slice(0);
        const second_slice = first_slice.slice(1);
        expect([...second_slice._iorder_value_iterator()]).toEqual([4, 5, 6, 7]);
      });

      it('3: qr decomposition', function(){
        // RESOLVED. No change required. Spec misunderstanding.
        const a = numts.arange(15).reshape(5, 3);
        const b = a.slice([0, null], [0, 1]);
        expect(b.shape).toEqual(new Uint32Array([5, 1]));
      });

    });
  });

  describe('reshape.', function () {
    it('array passed', function () {
      let start = numts.from_nested_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
      let reshaped = start.reshape(new Uint32Array([2, 2, 3]));
      let expected = numts.from_nested_array([
        [
          [0, 1, 2], [3, 4, 5],
        ], [
          [6, 7, 8], [9, 10, 11],
        ],
      ]);

      expect(reshaped.equals(expected)).toBe(true);
    });

    it('spread.', function () {
      const array = numts.arange(10).reshape(2, 5);
      expect(array.shape).toEqual(new Uint32Array([2, 5]));
      expect(array).toEqual(numts.from_nested_array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
      ], 'int32'));
    });
  });

  describe('drop_unit_dimensions.' ,function () {
    it('Simple test.', function () {
      const array = numts.arange(5).reshape([1, 5]);
      const dropped = array.drop_unit_dimensions();
      const expected = numts.arange(5);
      expect(expected.equals(dropped)).toBe(true);
    });

    it('Multiple dimensions.', function () {
      const array = numts.arange(25).reshape([1, 5, 1, 1, 5, 1]);
      const dropped = array.drop_unit_dimensions();
      const expected = numts.arange(25).reshape([5, 5]);
      expect(expected.equals(dropped)).toBe(true);
    })
  });
});

describe('Methods.', function () {

  describe('set', function() {
    it('single value', function() {
      let x = numts.arange(10).reshape(2, 5);
      x.s(-20, 0, 0);
      expect(x.g(0, 0)).toBe(-20);
    });

    it('slice.', function(){
      let x = numts.arange(10).reshape(2, 5);
      const replacement = numts.arange(10, 14).reshape(2, 2);
      x.s(replacement, [0, 2], [0, 2]);
      const values = [...x.slice([0,2], [0, 2])._iorder_value_iterator()];
      expect(values).toEqual([10, 11, 12, 13]);
    });

    // it('From failed matrix multiplication.', function() {
    //   let x = numts.zeros([2, 3, 4]);
    //   const a = numts.from_nested_array([
    //     [56, 62, 68, 74],
    //     [152, 174, 196, 218]
    //   ])
    // })
  });

});

describe('Unary methods.', function () {

  describe('Nonzero.', function () {
    it('simple.', function () {
      const array = numts.from_nested_array([
        [0, 1, 0], [2, 0, 1]
      ]);
      const expected = [
        new Uint32Array([0, 1]),
        new Uint32Array([1, 0]),
        new Uint32Array([1, 2]),
      ];

      expect(array.nonzero()).toEqual(expected);

    });
  });

  describe('Methods along axes.', function () {

    


  });

});

describe('Broadcasting.', function () {

  it('Broadcast on axis.', function () {
    let x = numts.arange(30).reshape([3, 2, 5]);
    let y = x.sum(1);
    const expected_data = [
      [5, 7, 9, 11, 13],
      [25, 27, 29, 31, 33],
      [45, 47, 49, 51, 53]
    ];

    const expected_array = numts.from_nested_array(expected_data, 'int32');
    expect(expected_array.equals(y)).toBe(true);
  });

  describe('From failures.', function() {
  }); 

  describe('_binary_broadcast.', function () {

  });
});

describe('Aggregation.', function () {

  let array = numts.arange(30).reshape([3, 2, 5]);

    describe('min.', function () {
      it('min over all.', function () {
        expect(array.min()).toBe(0);
      });

      it('min over 0.', function () {
        const expected = numts.from_nested_array([
          [0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9]
        ], 'int32');
        expect(array.min(0).equals(expected)).toBe(true);
      });
    });

    describe('max.', function () {
      it('max over all.', function () {
        expect(array.max()).toBe(29);
      });

      it('max over 0.', function () {
        const expected = numts.from_nested_array([
          [20, 21, 22, 23, 24],
          [25, 26, 27, 28, 29]
        ], 'int32');
        expect(array.max(0).equals(expected)).toBe(true);
      });
    });

  describe('all.', function() {

    it('No axis.', function() {
      const a = numts.arange(5).add(1);
      expect(a.all()).toBe(true);
    });

    it('Basic.', function() {
      const a = numts.from_nested_array([
        [0, 0, 1],
        [1, 1, 1]
      ], 'int32');
      expect(a.all()).toBe(false);
      expect(a.all(0).data).toEqual(new Int32Array([0, 0, 1]));
      expect(a.all(1).data).toEqual(new Int32Array([0, 1]));
    });
  });

  describe('any.', function() {
    it('Basic.', function() {
      const a = numts.from_nested_array([
        [0, 0, 1],
        [0, 1, 1]
      ], 'int32');
      expect(a.any()).toBe(true);
      expect(a.any(0).data).toEqual(new Int32Array([0, 1, 1]));
      expect(a.any(1).data).toEqual(new Int32Array([1, 1]));
    });
  });

  describe('cumsum.', function () {
    it('No axes.', function () {
      const input = numts.arange(12).reshape([3, 4]);
      const x = input.cumsum();
      const expected = numts.from_iterable([0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66], [12], 'int32');
      expect(x.equals(expected)).toBe(true);
    });

    it('2d, axis 1.', function () {
      const input = numts.arange(12).reshape([3, 4]);
      const result = input.cumsum(1);
      const expected = numts.from_nested_array([
        [0, 1, 3, 6],
        [4, 9, 15, 22],
        [8, 17, 27, 38]
      ], 'int32');
      expect(result.equals(expected)).toBe(true);
    });
  });

  describe('cumprod.', function () {
    it('No axes.', function () {
      const input = numts.arange(1, 13).reshape([3, 4]);
      const x = input.cumprod();
      const expected = numts.from_iterable([1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600], [12], 'int32');
      expect(x.equals(expected)).toBe(true);
    });

    it('2d, axis 1.', function () {
      const input = numts.arange(1, 13).reshape([3, 4]);
      const result = input.cumprod(1);
      const expected = numts.from_nested_array([
        [1, 2, 6, 24],
        [5, 30, 210, 1680],
        [9, 90, 990, 11880]
      ], 'int32');
      expect(result.equals(expected)).toBe(true);
    });
  });

  describe('variance.', function() {
    it('Basic.', function() {
      
    })
  });

  describe('mean.', function () {
    it('2d.', function () {
      const array = numts.arange(25).reshape([5, 5]);
      const mean = array.mean();
      expect(mean).toBeCloseTo(12.0);

    });
  });

  describe('stdev.', function () {
    it('2d.', function () {
      const array = numts.arange(25).reshape([5, 5]);
      const stdev = array.stdev();
      expect(stdev).toBeCloseTo(7.211);
    });
  });

  describe("sum.", function () {
    it("Simple.", function () {
      let x = numts.arange(30).reshape([3, 2, 5]);
      let y = x.sum(1);
      const expected_data = [
        [5, 7, 9, 11, 13],
        [25, 27, 29, 31, 33],
        [45, 47, 49, 51, 53]
      ];
  
      const expected_array = numts.from_nested_array(expected_data, 'int32');
      expect(expected_array.equals(y)).toBe(true);
    });
  });

  describe('reduce', function() {
    it('on slice.', function() {
      const a = numts.arange(30).reshape(5, 6);
      const b = a.slice([2, 4]);
      const res = b.reduce((x, y) => x + y);
      expect(res).toBe(210);
    })
  });
});

describe('Method constructors.', function() {

  describe('flatten.', function() {
    it('Simple test.', function() {
      let a = numts.arange(30).reshape(2, 3, 5);
      let b = a.flatten();
      let expected = numts.arange(30);
      expect(b.equals(expected)).toBe(true);
    })
    
  });

  describe('transpose.', function() {
    it('Two dimensions.', function() {
      const a = numts.arange(4).reshape(2, 2);
      const b = a.transpose();
      const expected = numts.from_nested_array([
        [0, 2],
        [1, 3]
      ], 'int32');
      
      expect(b.equals(expected)).toBe(true);
    });

    describe('Failures.', function() {
      it('Failed in QR decomp', function() {
        const a = numts.from_nested_array([[0], [1], [0.5345224838248488], [0.8017837257372732]]);
        const b = a.transpose();
        const expected = numts.from_nested_array([[0, 1, 0.5345224838248488, 0.8017837257372732]]);
        expect(b.equals(expected)).toBe(true);
      });
    });
  });

  describe('tril.', function() {
    it('Three dimensions.', function() {
      const a = numts.arange(30).reshape(3, 2, 5);
      const b = a.tril();
      const expected = numts.from_nested_array([
        [[0, 0, 0, 0, 0],
        [5, 6, 0, 0, 0]],
        [[10, 0, 0, 0, 0],
        [15, 16, 0, 0, 0]],
        [[20, 0, 0, 0, 0],
        [25, 26, 0, 0, 0]]
      ], 'int32');
      expect(b.equals(expected)).toBe(true);
    });
  });

  describe('triu.', function() {
    it('Three dimensions.', function() {
      const a = numts.arange(30).reshape(3, 2, 5);
      const b = a.triu();
      const expected = numts.from_nested_array([
        [[ 0,  1,  2,  3,  4],
        [ 0,  6,  7,  8,  9]],
        [[10, 11, 12, 13, 14],
        [ 0, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
        [ 0, 26, 27, 28, 29]]
      ], 'int32');
      expect(b.equals(expected)).toBe(true);
    });
  });
})

describe('Static methods.', () => {
    describe('Copy.', () => {

        describe('Previous breaks.', () => {
            it('householder_vector break. 2021-01-30.', () => {
                // Source: copy doesn't copy the initial offset. Issue #178.
                const a = numts.from_nested_array([
                    [1, 6,  11],
                    [2, 7, 12],
                    [3, 8, 13],
                    [4, 9, 14],
                    [5, 10, 15]
                ]);
                const s = a.slice([1, null], [0, 1]);
                const cp = tensor.copy(s);
                expect(s.equals(cp)).toBe(true);
            })
        })
    })
})