#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <cstdio>
#include <bitset>
#include <array>

#include <cstring>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <stdexcept>
#include <memory>
#define PCG_FORCE_EMULATED_128BIT_MATH
extern "C"{
    #include "pcg64.h"
}
#include "seed_sequence.h"

struct gf2Result {
    std::size_t rankVal = 0;
    std::bitset<128> depI, depJ;
};

gf2Result gf2RankSubset(const std::bitset<128> &I, const std::bitset<128> &J, const std::array<std::bitset<128>, 128> &M) {
    const std::bitset<128> &selected_inputs = I;
    const std::bitset<128> &selected_outputs = J;
    const std::array<std::bitset<128>, 128> &outputs = M;
    std::array<std::bitset<128>, 256> rows;
    int selected_count = 0;
    int candidate_count = 0;
    for (int i = 0; i < 128; i++) {
        int row = selected_inputs[i] ? selected_count++ : 255 - candidate_count++;
        rows[row][i] = 1;
    }
    for (int i = 0; i < 128; i++) {
        int row = selected_outputs[i] ? selected_count++ : 255 - candidate_count++;
        rows[row] = outputs[i];
    }

    // std::printf("Native top =\n");
    // for (int row = 0; row < selected_count; row++) {
    //     for (int i = 0; i < 128; i++) {
    //         int val = rows[row][i];
    //         std::printf("%d", val);
    //     }
    //     std::printf("\n");
    // }

    int used_rows = 0;

    for (int col = 0; col < 128; col++) {
        int main_row = -1;

        for (int row = used_rows; row < selected_count; row++) {
            if (rows[row][col]) {
                main_row = row;
                break;
            }
        }

        if (main_row == -1) {
            continue;
        }

        if (main_row != used_rows) {
            rows[used_rows] ^= rows[main_row];
        }

        std::bitset<128> main_row_val = rows[used_rows];

        for (int row = used_rows + 1; row < 256; row++) {
            if (rows[row][col]) {
                rows[row] ^= main_row_val;
            }
        }

        used_rows += 1;
    }

    std::bitset<128> dependant_inputs;
    std::bitset<128> dependant_outputs;
    candidate_count = 0;
    for (int i = 0; i < 128; i++) {
        if (selected_inputs[i]) continue;
        int row = 255 - candidate_count++;
        dependant_inputs[i] = rows[row].none();
    }
    for (int i = 0; i < 128; i++) {
        if (selected_outputs[i]) continue;
        int row = 255 - candidate_count++;
        dependant_outputs[i] = rows[row].none();
    }

    return { (size_t) used_rows, dependant_inputs, dependant_outputs };
}

template<size_t _Bits>
std::bitset<_Bits> ndarray_to_bitset(PyArrayObject *arr) {
    std::bitset<_Bits> bitset;
    bool *data = (bool*) PyArray_DATA(arr);
    for (int i = 0; i < _Bits; i++) {
        bitset[i] = data[i];
    }
    return bitset;
}

template<size_t _Rows, size_t _Bits>
std::array<std::bitset<_Bits>, _Rows> ndarray_to_bitset_2d(PyArrayObject *arr) {
    std::array<std::bitset<_Bits>, _Rows> bitset;
    bool *data = (bool*) PyArray_DATA(arr);
    for (int row = 0; row < _Rows; row++) {
        for (int i = 0; i < _Bits; i++) {
            // bitset[row][i] = data[row * _Bits + i];
            bitset[row][i] = data[i * _Rows + row];
        }
    }
    return bitset;
}

template<size_t _Bits>
PyArrayObject *bitset_to_ndarray(std::bitset<_Bits> bitset) {
    npy_intp dims[] = { _Bits };
    PyArrayObject *arr = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_BOOL);
    bool *data = (bool*) PyArray_DATA(arr);
    for (int i = 0; i < _Bits; i++) {
        data[i] = bitset[i];
    }
    return arr;
}

static PyObject *
gf2_rank_subset(PyObject *self, PyObject *args) {
    PyArrayObject *I;
    PyArrayObject *J;
    PyArrayObject *M;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &I, &PyArray_Type, &J, &PyArray_Type, &M)) {
        goto error;
    }

    if (PyArray_TYPE(I) != NPY_BOOL || PyArray_NDIM(I) != 1 || PyArray_DIM(I, 0) != 128) goto none;
    if (PyArray_TYPE(J) != NPY_BOOL || PyArray_NDIM(J) != 1 || PyArray_DIM(J, 0) != 128) goto none;
    if (PyArray_TYPE(M) != NPY_BOOL || PyArray_NDIM(M) != 2 || PyArray_DIM(M, 0) != 128 || PyArray_DIM(M, 1) != 128) goto none;

    // std::printf("M stride 0: %zd\n", PyArray_STRIDE(M, 0));
    // std::printf("M stride 1: %zd\n", PyArray_STRIDE(M, 1));

    {
        auto I_bitset = ndarray_to_bitset<128>(I);
        auto J_bitset = ndarray_to_bitset<128>(J);
        auto M_bitset = ndarray_to_bitset_2d<128, 128>(M);

        // std::printf("Native input:\n");
        // std::printf("I = ");
        // for (int i = 0; i < 128; i++) {
        //     int val = I_bitset[i];
        //     std::printf("%d", val);
        // }
        // std::printf("\n");
        // std::printf("J = ");
        // for (int i = 0; i < 128; i++) {
        //     int val = J_bitset[i];
        //     std::printf("%d", val);
        // }
        // std::printf("\n");
        // std::printf("M =\n");
        // for (int row = 0; row < 128; row++) {
        //     for (int i = 0; i < 128; i++) {
        //         int val = M_bitset[row][i];
        //         std::printf("%d", val);
        //     }
        //     std::printf("\n");
        // }

        auto res = gf2RankSubset(I_bitset, J_bitset, M_bitset);

        // std::printf("Native output:\n");
        // std::printf("Rank = %zu\n", res.rankVal);
        // std::printf("Dep I = ");
        // for (int i = 0; i < 128; i++) {
        //     int val = res.depI[i];
        //     std::printf("%d", val);
        // }
        // std::printf("\n");
        // std::printf("Dep J = ");
        // for (int i = 0; i < 128; i++) {
        //     int val = res.depJ[i];
        //     std::printf("%d", val);
        // }
        // std::printf("\n");

        PyObject *t = PyTuple_New(3);
        PyTuple_SetItem(t, 0, PyLong_FromLong(res.rankVal));
        PyTuple_SetItem(t, 1, (PyObject *) bitset_to_ndarray(res.depI));
        PyTuple_SetItem(t, 2, (PyObject *) bitset_to_ndarray(res.depJ));
        return t;
    }

    none:
    return Py_None;

    error:
    return NULL;
}

// PCG-64 implementation
// Based on the official PCG implementation

// Global RNG state storage - we'll use a dictionary to store multiple RNG states
static PyObject* rng_states_dict = NULL;

static void pcg64_state_destructor(PyObject* capsule) {
    pcg64_state* rng = (pcg64_state*)PyCapsule_GetPointer(capsule, "pcg64_state");
    if (rng) {
        free(rng);
    }
}

static PyObject* init_rng_state(PyObject* self, PyObject* args) {
    unsigned long long seed;
    PyObject* rng_id_obj;
    
    // Parse arguments: seed and RNG identifier
    if (!PyArg_ParseTuple(args, "KO", &seed, &rng_id_obj)) {
        return NULL;
    }
    
    // Create RNG state
    pcg64_state rng;
    rng.pcg_state = (pcg64_random_t*)malloc(sizeof(pcg64_random_t));
    memset(rng.pcg_state, 0, sizeof(pcg64_random_t));
    SeedState seed_state;
    init_seed_state(seed, &seed_state);
    std::vector<uint64_t> state_array;
    generate_state(seed_state, state_array, 4);
    pcg64_set_seed(&rng, &state_array[0], &state_array[2]);
    uint64_t stlow = rng.pcg_state->state.low;
    uint64_t sthigh = rng.pcg_state->state.high;
    uint64_t inclow = rng.pcg_state->inc.low;
    uint64_t inchigh = rng.pcg_state->inc.high;
    uint64_t *arr = (uint64_t*) malloc(sizeof(uint64_t)*4);
    arr[0] = sthigh;
    arr[1] = stlow;
    arr[2] = inchigh;
    arr[3] = inclow;
    pcg64_set_state(&rng, arr, 0, 0);

    
    
    // Create a PyCapsule to hold the RNG state
    PyObject* rng_capsule = PyCapsule_New(&rng, "pcg64_state", pcg64_state_destructor);
    if (!rng_capsule) {
        free(&rng);
        return NULL;
    }
    
    // Initialize the global dictionary if it doesn't exist
    if (!rng_states_dict) {
        rng_states_dict = PyDict_New();
        if (!rng_states_dict) {
            Py_DECREF(rng_capsule);
            free(&rng);
            return NULL;
        }
    }
    
    // Store the RNG state in the dictionary
    if (PyDict_SetItem(rng_states_dict, rng_id_obj, rng_capsule) < 0) {
        Py_DECREF(rng_capsule);
        free(&rng);
        return NULL;
    }
    
    Py_DECREF(rng_capsule);  // Dict now owns a reference
    Py_RETURN_NONE;
}

static PyObject* get_rng_state(PyObject* self, PyObject* args) {
    PyObject* rng_id_obj;
    
    // Parse arguments: RNG identifier
    if (!PyArg_ParseTuple(args, "O", &rng_id_obj)) {
        return NULL;
    }
    
    // Get the RNG state from the dictionary
    if (!rng_states_dict) {
        PyErr_SetString(PyExc_RuntimeError, "RNG states not initialized");
        return NULL;
    }
    
    PyObject* rng_capsule = PyDict_GetItem(rng_states_dict, rng_id_obj);
    if (!rng_capsule) {
        PyErr_SetString(PyExc_KeyError, "RNG state not found for the given identifier");
        return NULL;
    }
    
    pcg64_state* rng = (pcg64_state*)PyCapsule_GetPointer(rng_capsule, "pcg64_state");
    if (!rng) {
        return NULL;  // Exception already set by PyCapsule_GetPointer
    }
    
    // Create a tuple to return the RNG state
    PyObject* state_tuple = PyTuple_New(4);
    //void pcg64_get_state(pcg64_state *state, uint64_t *state_arr, int *has_uint32,
    //uint32_t *uinteger);
    uint64_t output_arr[4];
    pcg64_get_state(rng, output_arr, 0, 0);
    
    return state_tuple;
}

/**
 * Set the state of an RNG
 */
static PyObject* set_rng_state(PyObject* self, PyObject* args) {
    PyObject* rng_id_obj;
    PyObject* state_tuple;
    
    // Parse arguments: RNG identifier and state tuple
    if (!PyArg_ParseTuple(args, "OO", &rng_id_obj, &state_tuple)) {
        return NULL;
    }
    
    // Check that state_tuple is a tuple of 4 integers
    if (!PyTuple_Check(state_tuple) || PyTuple_Size(state_tuple) != 4) {
        PyErr_SetString(PyExc_TypeError, "State must be a tuple of 4 integers");
        return NULL;
    }
    
    // Get the RNG state from the dictionary
    if (!rng_states_dict) {
        PyErr_SetString(PyExc_RuntimeError, "RNG states not initialized");
        return NULL;
    }
    
    PyObject* rng_capsule = PyDict_GetItem(rng_states_dict, rng_id_obj);
    if (!rng_capsule) {
        PyErr_SetString(PyExc_KeyError, "RNG state not found for the given identifier");
        return NULL;
    }
    
    pcg64_state* rng = (pcg64_state*)PyCapsule_GetPointer(rng_capsule, "pcg64_state");
    if (!rng) {
        return NULL;  // Exception already set by PyCapsule_GetPointer
    }
    
    // Set the RNG state
    uint64_t arr[4];
    arr[0] = PyLong_AsUnsignedLongLong(PyTuple_GetItem(state_tuple, 0));
    arr[1] = PyLong_AsUnsignedLongLong(PyTuple_GetItem(state_tuple, 1));
    arr[2] = PyLong_AsUnsignedLongLong(PyTuple_GetItem(state_tuple, 2));
    arr[3] = PyLong_AsUnsignedLongLong(PyTuple_GetItem(state_tuple, 3));

   
    pcg64_set_state(rng, arr, 0, 0);
    
    Py_RETURN_NONE;
}
// Fisher-Yates shuffle implementation for sampling without replacement
void shuffle(std::vector<int>& array, int n, pcg64_state* rng) {
    for (int i = 0; i < n; i++) {
        // Generate random index between i and array.size()-1 (inclusive)
        uint64_t range = array.size() - i;
        uint64_t random_val = pcg64_next64(rng);
        int j = i + (random_val % range);
        
        // Swap elements at i and j
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

static PyObject* generate_random_binary(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* inner_bound_obj = NULL;
    PyObject* outer_bound_obj = NULL;
    PyObject* rng_id_obj = NULL;
    int n_ones = 0;
    
    static char* kwlist[] = {"n_ones", "inner_bound", "outer_bound", "rng", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOOO", kwlist, 
                                    &n_ones, &inner_bound_obj, &outer_bound_obj, &rng_id_obj)) {
        return NULL;
    }
    
    // Convert inner_bound and outer_bound to numpy arrays (accepting any integer type)
    PyArrayObject* inner_bound_array = (PyArrayObject*)PyArray_FROM_OTF(
        inner_bound_obj, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    if (!inner_bound_array) {
        return NULL;
    }
    
    PyArrayObject* outer_bound_array = (PyArrayObject*)PyArray_FROM_OTF(
        outer_bound_obj, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    if (!outer_bound_array) {
        Py_DECREF(inner_bound_array);
        return NULL;
    }
    
    // Get the RNG state from the dictionary
    if (!rng_states_dict) {
        PyErr_SetString(PyExc_RuntimeError, "RNG states not initialized");
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;
    }
    
    PyObject* rng_capsule = PyDict_GetItem(rng_states_dict, rng_id_obj);
    if (!rng_capsule) {
        PyErr_SetString(PyExc_KeyError, "RNG state not found for the given identifier");
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;
    }
    
    pcg64_state* rng = (pcg64_state*)PyCapsule_GetPointer(rng_capsule, "pcg64_state");
    if (!rng) {
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;  // Exception already set by PyCapsule_GetPointer
    }
    
    // Check that the two bounds arrays have equal length
    npy_intp L = PyArray_DIM(inner_bound_array, 0);
    if (L != PyArray_DIM(outer_bound_array, 0)) {
        PyErr_SetString(PyExc_AssertionError, "Bound arrays must have the same length.");
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;
    }
    
    // Get pointers to the data
    long* inner_bound_data = (long*)PyArray_DATA(inner_bound_array);
    long* outer_bound_data = (long*)PyArray_DATA(outer_bound_array);
    
    // Check for conflicting requirements
    int forced_ones = 0;
    int forced_zeros = 0;
    
    for (npy_intp i = 0; i < L; i++) {
        if (inner_bound_data[i] == 1 && outer_bound_data[i] == 0) {
            PyErr_SetString(PyExc_AssertionError, 
                           "Conflicting requirements: inner_bound forces one where outer_bound forces zero.");
            Py_DECREF(inner_bound_array);
            Py_DECREF(outer_bound_array);
            return NULL;
        }
        
        if (inner_bound_data[i] == 1) {
            forced_ones++;
        }
        
        if (outer_bound_data[i] == 0) {
            forced_zeros++;
        }
    }
    
    // Check that the requested n_ones is possible
    int max_possible_ones = L - forced_zeros;
    if (forced_ones > n_ones || n_ones > max_possible_ones) {
        PyErr_SetString(PyExc_AssertionError, 
                       "n_ones not possible given the forced ones and zeros.");
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;
    }
    
    // Create output array (using int64 to match input arrays)
    npy_intp dims[1] = {L};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_LONG);
    if (!output_array) {
        Py_DECREF(inner_bound_array);
        Py_DECREF(outer_bound_array);
        return NULL;
    }
    
    long* output_data = (long*)PyArray_DATA(output_array);
    
    // Fill forced positions and collect free positions
    std::vector<int> free_indices;
    
    for (npy_intp i = 0; i < L; i++) {
        if (inner_bound_data[i] == 1) {
            output_data[i] = 1;
        } else if (outer_bound_data[i] == 0) {
            output_data[i] = 0;
        } else {
            // This is a free position
            free_indices.push_back(i);
        }
    }
    
    // Calculate how many additional ones need to be placed
    int additional_ones = n_ones - forced_ones;
    
    // Randomly choose free positions to set to 1
    if (additional_ones > 0) {
        // Use Fisher-Yates shuffle to select indices
        shuffle(free_indices, additional_ones, rng);
        
        // Set selected indices to 1
        for (int i = 0; i < additional_ones; i++) {
            output_data[free_indices[i]] = 1;
        }
        
        // Set remaining free positions to 0
        for (int i = additional_ones; i < free_indices.size(); i++) {
            output_data[free_indices[i]] = 0;
        }
    } else {
        // Set all free positions to 0
        for (int i = 0; i < free_indices.size(); i++) {
            output_data[free_indices[i]] = 0;
        }
    }
    
    Py_DECREF(inner_bound_array);
    Py_DECREF(outer_bound_array);
    
    return PyArray_Return(output_array);
}

static PyMethodDef methods[] = {
    {"gf2_rank_subset", gf2_rank_subset, METH_VARARGS, NULL},
    {"init_rng_state", init_rng_state, METH_VARARGS,
        "Initialize a new RNG state with a given seed and store it with an identifier."},
    {"generate_random_binary", (PyCFunction)generate_random_binary, 
        METH_VARARGS | METH_KEYWORDS, 
        "Generate a random binary array with constraints on the number of ones and positions."},
    {"get_rng_state", get_rng_state, METH_VARARGS,
        "Get the current state of an RNG."},
    {"set_rng_state", set_rng_state, METH_VARARGS,
        "Set the state of an RNG."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "native",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC
PyInit_native(void)
{
    import_array();

    // added in numpy 2.0
    // if (PyArray_ImportNumPyAPI() != 0) {
    //     goto error;
    // }

    return PyModule_Create(&module);
}