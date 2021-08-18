const call_python = require('../call_python');
const tensor = require('../../numts/tensor').tensor;
const numts = require('../../numts/numts');
const helpers = require('./helpers');


describe('Basic operations.', () => {

  function check_op(op) {
    const f = (a, b) => {
        // console.log('a and b')
        // console.log(a)
        // console.log(b)
        const numts_value = tensor[op](a, b);
        // console.log("numts value")
        // console.log(numts_value);
        const a_string = JSON.stringify(a.to_json());
        const b_string = JSON.stringify(b.to_json());
        const py_str = call_python.call_python(op, [a_string, b_string])
        const py_value = tensor.from_json(JSON.parse(py_str))

        // console.log(py_value);
        return numts_value.is_close(py_value)
    };
    return f;
  }

  test('Array addition.', () => {
    helpers.check_arrays(check_op('_add'));
  });
});
