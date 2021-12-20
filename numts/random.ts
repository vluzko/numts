import {tensor} from "./tensor";
import {from_iterable} from './tensor_core/constructors';
import {indexing} from "./tensor_core/indexing";
import {Shape} from "./types";

/**
 * 
 * @param lower - The lower bound of the distribution. Defaults to 0.
 * @param upper - The upper bound of the distribution. Defaults to 1.
 * @param shape - The shape of the array to generate. Defaults to [1].
 */
export function randint(lower: number = 0.0, upper: number = 1.0, shape: Shape = [1]): tensor {
  const scale_factor = upper - lower;
  const size = indexing.compute_size(shape);

  let iter = {
    [Symbol.iterator]: function*() {
      for (let i = 0; i < size; i++) {
        yield Math.floor((Math.random() * scale_factor) + lower);
      }
    }
  };

  return from_iterable(iter, shape, "int32");
}

/**
 * Create an array of uniformly distributed values.
 * @param lower - The lower bound of the distribution. Defaults to 0.
 * @param upper - The upper bound of the distribution. Defaults to 1.
 * @param shape - The shape of the array to generate. Defaults to [1].
 */
export function uniform(lower: number = 0.0, upper: number = 1.0, shape: Shape = [1]): tensor {

  const scale_factor = upper - lower;
  const size = indexing.compute_size(shape);

  let iter = {
    [Symbol.iterator]: function*() {
      for (let i = 0; i < size; i++) {
        yield (Math.random() * scale_factor) + lower;
      }
    }
  };

  return from_iterable(iter, shape, "float64");
}

/**
 * Create an array of normally distributed values.
 * Uses the Ziggurat algorithm.
 * @param mean  - The mean of the distribution. Defaults to 0.
 * @param stdev - The standard deviation of the distribution. Defaults to 1.
 * @param shape - The shape of the array to generate. Defaults to [1].
 */
export function normal(mean: number = 0.0, stdev: number = 1.0, shape: Shape = [1]): tensor {
  const size = indexing.compute_size(shape);

  let iter = {
    [Symbol.iterator]: function*() {
      for (let i = 0; i < size; i++) {
        yield ziggurat_normal() * stdev + mean;
      }
    }
  };

  return from_iterable(iter, shape, "float64");
}

export function binomial() {

}

export function beta () {

}

export function exponential() {

}

export function gamma() {
    
}

export function laplace() {

}

export function multinomial() {

}

export function pareto() {

}

export function poisson() {

}

export function power() {

}


/**
 * Generate a normally distributed random value, using the Ziggurat algorithm.
 */
function ziggurat_normal(): number {
  const layers = 128;
  // 50% chance of positive value, 50% chance of negative value.
  const right_hand_side = Math.random() < 0.5;
  let x : number;
  let done = false;
  while (!done) {
    const layer = Math.floor(Math.random() * layers);
    const u_0 = Math.random();
    x = u_0 * xtab_normal[layer];
    // If our generated value is below the corresponding ratio, keep it.
    if (x < ktab_normal[layer]) {
      done = true;
    } else {
      let y: number;
      // If we are not looking at the last layer, everything is easier.
      if (layer < layers - 1) {
        const y_0 = ytab_normal[layer];
        const y_1 = ytab_normal[layer + 1];
        y = y_1 + (y_0 - y_1) * Math.random();
      } else {
        // If we are looking at the last layer, it's more annoying.
        x = RIGHT_MOST - Math.log(1.0 - Math.random()) / RIGHT_MOST;
        y = Math.exp(-RIGHT_MOST * (x - 0.5 * RIGHT_MOST)) * Math.random();
      }
      if (y < Math.exp(-0.5 * x * x)) {
        done = true;
      }
    }
  }
  return right_hand_side ? x : -x;
}

// The x-coordinate of the start of the last normal layer.
const RIGHT_MOST = 3.44428647676;

// Tabulated values of y[i] for the normal distribution. 
const ytab_normal = new Float64Array([
  1.        , 0.96359862, 0.93628081, 0.9130411 , 0.89227851,
  0.87323936, 0.85549641, 0.83877893, 0.82290208, 0.80773274,
  0.79317105, 0.77913973, 0.76557744, 0.75243446, 0.73966979,
  0.72724912, 0.71514338, 0.70332765, 0.69178038, 0.68048277,
  0.6694183 , 0.65857234, 0.64793188, 0.63748525, 0.62722199,
  0.61713261, 0.60720852, 0.59744188, 0.58782553, 0.57835291,
  0.56901798, 0.55981517, 0.55073932, 0.54178566, 0.53294974,
  0.52422743, 0.51561489, 0.50710849, 0.49870487, 0.49040085,
  0.48219348, 0.47407994, 0.4660576 , 0.45812397, 0.45027671,
  0.4425136 , 0.43483254, 0.42723153, 0.41970869, 0.41226223,
  0.40489045, 0.39759172, 0.39036451, 0.38320736, 0.37611886,
  0.36909769, 0.36214259, 0.35525233, 0.34842577, 0.3416618 ,
  0.33495938, 0.32831749, 0.32173517, 0.31521151, 0.30874564,
  0.3023367 , 0.29598391, 0.2896865 , 0.28344373, 0.27725491,
  0.27111938, 0.26503649, 0.25900565, 0.25302628, 0.24709783,
  0.24121978, 0.23539164, 0.22961293, 0.22388322, 0.21820208,
  0.21256912, 0.20698398, 0.20144631, 0.19595578, 0.19051209,
  0.18511498, 0.1797642 , 0.1744595 , 0.1692007 , 0.16398761,
  0.15882008, 0.15369797, 0.14862119, 0.14358966, 0.13860332,
  0.13366216, 0.12876619, 0.12391544, 0.11910999, 0.11434994,
  0.10963544, 0.10496667, 0.10034386, 0.09576727, 0.09123724,
  0.08675413, 0.08231838, 0.07793049, 0.07359105, 0.06930071,
  0.06506023, 0.06087048, 0.05673245, 0.05264727, 0.04861626,
  0.04464094, 0.04072307, 0.03686473, 0.03306838, 0.029337  ,
  0.02567418, 0.02208444, 0.01857352, 0.01514906, 0.01182165,
  0.00860719, 0.00553245, 0.00265435
]);

// Tabulated values of x[i] / x[i+1], for the normal distribution.
const ktab_normal = new Uint32Array([
  0.0       , 0.75046086, 0.85071641, 0.89341038, 0.91699266,
  0.93191916, 0.94220406, 0.9497152 , 0.95543826, 0.95994204,
  0.96357745, 0.9665727 , 0.96908259, 0.97121567, 0.97305048,
  0.97464508, 0.9760434 , 0.97727931, 0.97837919, 0.97936404,
  0.98025084, 0.98105329, 0.98178256, 0.9824481 , 0.98305762,
  0.98361778, 0.98413414, 0.98461145, 0.98505372, 0.98546457,
  0.98584706, 0.98620385, 0.98653722, 0.98684931, 0.98714191,
  0.98741657, 0.98767477, 0.98791784, 0.98814684, 0.98836285,
  0.98856676, 0.98875934, 0.98894149, 0.98911375, 0.98927683,
  0.9894312 , 0.98957741, 0.98971599, 0.98984724, 0.98997164,
  0.99008948, 0.99020112, 0.99030685, 0.99040699, 0.9905017 ,
  0.99059123, 0.99067581, 0.99075562, 0.99083084, 0.99090159,
  0.99096805, 0.99103034, 0.99108863, 0.99114293, 0.99119335,
  0.99124002, 0.991283  , 0.99132228, 0.99135804, 0.99139023,
  0.99141884, 0.99144405, 0.99146569, 0.99148393, 0.99149859,
  0.9915098 , 0.99151748, 0.9915216 , 0.99152207, 0.99151886,
  0.99151194, 0.99150109, 0.99148637, 0.9914676 , 0.99144459,
  0.99141729, 0.99138546, 0.99134892, 0.9913075 , 0.99126089,
  0.99120891, 0.99115115, 0.99108732, 0.99101704, 0.99093986,
  0.99085534, 0.99076289, 0.99066186, 0.99055159, 0.99043125,
  0.99029994, 0.99015659, 0.99000001, 0.98982877, 0.98964125,
  0.98943549, 0.98920929, 0.98895997, 0.98868424, 0.98837835,
  0.98803759, 0.98765612, 0.98722678, 0.98674047, 0.98618567,
  0.98554724, 0.98480505, 0.9839319 , 0.98288965, 0.98162335,
  0.98005044, 0.97804093, 0.97537607, 0.97165602, 0.96605647,
  0.95653498, 0.93604612, 0.92225826
]);

// Tabulated values of x[i] for the normal distribution.
const xtab_normal = new Float64Array([
  0.27232494, 0.36287693, 0.42655452, 0.47744522, 0.52066416,
  0.55870097, 0.59297235, 0.62436858, 0.65348917, 0.68075897,
  0.70649116, 0.73092395, 0.75424318, 0.77659697, 0.7981055 ,
  0.81886779, 0.83896654, 0.8584716 , 0.87744262, 0.89593098,
  0.9139813 , 0.93163267, 0.94891952, 0.9658724 , 0.98251855,
  0.9988824 , 1.01498599, 1.03084925, 1.04649034, 1.06192583,
  1.07717093, 1.09223961, 1.10714482, 1.12189853, 1.1365119 ,
  1.15099534, 1.16535858, 1.17961079, 1.19376057, 1.20781608,
  1.22178503, 1.23567476, 1.24949226, 1.26324421, 1.27693702,
  1.29057684, 1.30416961, 1.31772106, 1.33123675, 1.34472209,
  1.35818234, 1.37162264, 1.38504804, 1.39846349, 1.41187388,
  1.42528402, 1.43869869, 1.45212264, 1.46556059, 1.47901725,
  1.49249736, 1.50600566, 1.5195469 , 1.5331259 , 1.54674753,
  1.56041672, 1.57413848, 1.58791792, 1.60176024, 1.61567079,
  1.62965503, 1.64371858, 1.65786725, 1.67210702, 1.68644407,
  1.70088483, 1.71543598, 1.73010446, 1.74489752, 1.75982275,
  1.77488809, 1.7901019 , 1.80547294, 1.82101049, 1.83672432,
  1.85262479, 1.8687229 , 1.88503034, 1.90155958, 1.91832391,
  1.93533761, 1.95261597, 1.97017545, 1.98803382, 2.00621028,
  2.02472566, 2.04360264, 2.06286594, 2.08254262, 2.10266241,
  2.12325806, 2.14436584, 2.16602598, 2.18828343, 2.21118851,
  2.23479794, 2.25917595, 2.28439571, 2.31054111, 2.33770904,
  2.36601222, 2.395583  , 2.4265782 , 2.45918573, 2.49363349,
  2.53020182, 2.56924117, 2.61119805, 2.65665423, 2.70638851,
  2.76147863, 2.82347954, 2.89475978, 2.97920204, 3.08387964,
  3.22401118, 3.44428648, 3.73462237
]);