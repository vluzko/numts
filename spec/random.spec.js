const random = require("../numts/random");
const helpers = require("./helpers");


describe("Uniform distribution.", function() {
  describe("Shape tests", function() {
    it("One dimensional test", function(){
      let test_1 = random.uniform();
      expect(test_1.shape).toEqual(new Uint32Array([1]));
    });

    it("Two dimensional test", function() {
      let array = random.uniform(0, 1, [2, 3]);
      expect(array.shape).toEqual(new Uint32Array([2, 3]));
    });

    it("n-dimensional", function() {
      let shape = helpers.random_shape(5);
      let array = random.uniform(0, 1, shape);
      expect(array.shape).toEqual(shape);
    })
  });

  describe("Distribution tests.", function() {
    it("Full range test.", function() {
      const lower = Number.MIN_VALUE;
      const upper = Number.MAX_VALUE;
      const array = random.uniform(lower, upper, [10000]);
      expect(array.max()).toBeLessThan(upper);
      expect(array.min()).toBeGreaterThan(lower - 1);
    });

  });
});

describe("Normal distribution.", function() {
  describe("Distribution tests.", function() {

    it("Simple test", function() {
      const mean = random.uniform(-10, 10, [1]).g(0);
      const stdev = random.uniform(0, 10, [1]).g(0);
      const array = random.normal(mean, stdev, [10000]);
      const mean_diff = Math.abs(mean - array.mean());
      const stdev_diff = Math.abs(stdev - array.stdev());
      expect(mean_diff / mean).toBeLessThan(0.01);
      expect(stdev_diff / stdev).toBeLessThan(0.1);
    });
  });
});

describe("Discrete uniform distribution.", function() {
  describe("Distribution tests.", function() {
    it("Simple test.", function() {
      const lower = Math.floor(random.uniform(-100, 100).g(0));
      const upper = lower + Math.floor(random.uniform(1, 100).g(0));
      const array = random.randint(lower, upper, [10000]);
      expect(array.max()).toBeLessThan(upper);
      expect(array.min()).toBeGreaterThan(lower - 1);
    });
  });
});
