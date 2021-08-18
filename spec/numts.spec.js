let tensor = require("../numts/tensor");
let numts = require("../numts/numts");
let _ = require("lodash");


// TODO: Randomly generated tests.
// TODO: MAX_INT tests
describe("Constructors and factories.", function () {
  describe("Shapes.", function () {

    class Sample {
      constructor(input_array, input_shape, expected) {
        this.input_array = input_array;
        this.input_shape = input_shape;
        this.expected = expected;
      }
    }

    // TODO: Tests for wrong input types.
    const passing_samples = new Map([
      ["null", new Sample([], null, {
        "shape": new Uint32Array([0]),
        "length": 0,
      })],
      ["flat", new Sample([1, 2, 3, 4], [4], {
        "shape": new Uint32Array([4]),
        "length": 4,
      })],
      ["two-dimensional", new Sample([1, 2, 3, 4, 5, 6], [2, 3], {
        "shape": new Uint32Array([2, 3]),
        "length": 6,
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

  });

  describe("Specialized factories.", function () {
    it("zeros.", function () {
      expect(numts.zeros([2, 2]).data).toEqual(new Float64Array([0, 0, 0, 0]));
      expect(numts.zeros(4).data).toEqual(new Float64Array([0, 0, 0, 0]));
    });

    it("ones.", function () {
      expect(numts.ones([2, 2]).data).toEqual(new Float64Array([1, 1, 1, 1]));
      expect(numts.ones(4).data).toEqual(new Float64Array([1, 1, 1, 1]));
    });

    it("filled.", function () {
      expect(numts.filled(-1, [2, 2]).data).toEqual(new Float64Array([-1, -1, -1, -1]));
      expect(numts.filled(10, 4).data).toEqual(new Float64Array([10, 10, 10, 10]));
    });

  });
});

describe("Unary methods.", function () {

  describe("Nonzero.", function () {
    it("simple.", function () {
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

  describe("Methods along axes.", function () {

    let array = numts.arange(30).reshape([3, 2, 5]);

    describe("min.", function () {
      it("min over all.", function () {
        expect(array.min()).toBe(0);
      });

      it("min over 0.", function () {
        const expected = numts.from_nested_array([
          [0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9]
        ], "int32");
        expect(array.min(0).equals(expected)).toBe(true);
      });
    });

    describe("max.", function () {
      it("max over all.", function () {
        expect(array.max()).toBe(29);
      });

      it("max over 0.", function () {
        const expected = numts.from_nested_array([
          [20, 21, 22, 23, 24],
          [25, 26, 27, 28, 29]
        ], "int32");
        expect(array.max(0).equals(expected)).toBe(true);
      });
    });

    describe("mean.", function () {

    });
  });

});
