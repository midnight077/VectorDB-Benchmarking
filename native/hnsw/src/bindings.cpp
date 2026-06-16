#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/distances_simd.hpp"
#include "hnsw/hnsw_index.hpp"

namespace py = pybind11;

namespace {

template <class Dist>
void bind_index(py::module_& m, const char* name) {
    using Index = hnsw::HnswIndex<Dist>;

    py::class_<Index>(m, name)
        .def(py::init<int, int, int, uint64_t>(), py::arg("dim"), py::arg("M"),
             py::arg("ef_construction"), py::arg("seed") = 42)
        .def(
            "add_points",
            [](Index& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               bool parallel, int num_threads) {
                auto buf = data.request();
                if (buf.ndim != 2) throw std::runtime_error("add_points: data must be 2D (n, dim)");
                if (static_cast<int>(buf.shape[1]) != self.dim()) {
                    throw std::runtime_error("add_points: dim mismatch");
                }
                const int n = static_cast<int>(buf.shape[0]);
                const float* ptr = static_cast<const float*>(buf.ptr);
                py::gil_scoped_release release;
                self.add_points(ptr, n, parallel, num_threads);
            },
            py::arg("data"), py::arg("parallel") = false, py::arg("num_threads") = 1)
        .def("set_ef", &Index::set_ef, py::arg("ef_search"))
        .def(
            "search",
            [](const Index& self, py::array_t<float, py::array::c_style | py::array::forcecast> query,
               int k) {
                auto buf = query.request();
                if (buf.ndim != 1 || static_cast<int>(buf.shape[0]) != self.dim()) {
                    throw std::runtime_error("search: query must be 1D with length == dim");
                }
                std::vector<int> out(static_cast<size_t>(k));
                const float* ptr = static_cast<const float*>(buf.ptr);
                {
                    py::gil_scoped_release release;
                    self.search(ptr, k, out.data());
                }
                return out;
            },
            py::arg("query"), py::arg("k"))
        .def("size", &Index::size)
        .def("dim", &Index::dim)
        .def("index_memory_bytes", &Index::index_memory_bytes)
        .def("save", &Index::save, py::arg("path"))
        .def("load", &Index::load, py::arg("path"));
}

}  // namespace

PYBIND11_MODULE(_vdbhnsw, m) {
    m.doc() = "Custom HNSW (vdbbench native extension)";

    bind_index<hnsw::ScalarL2>(m, "HnswScalarL2");
    bind_index<hnsw::ScalarIP>(m, "HnswScalarIP");

    // Phase B: SIMD distance kernels, still built serially (no OpenMP yet).
    bind_index<hnsw::SimdL2>(m, "HnswFastL2");
    bind_index<hnsw::SimdIP>(m, "HnswFastIP");
}
