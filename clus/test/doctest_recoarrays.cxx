#include "WireCellAux/TensorDMpointtree.h"
#include "WireCellClus/Facade_Blob.h"
#include "WireCellClus/Facade_Cluster.h"
#include "WireCellClus/Facade_Grouping.h"
#include "WireCellClus/Facade_Util.h"
#include "WireCellUtil/PointTesting.h"
#include "WireCellUtil/PointTree.h"
#include "WireCellUtil/Stream.h"
#include "WireCellUtil/doctest.h"

#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace WireCell;
using namespace WireCell::PointCloud;
using namespace WireCell::PointCloud::Facade;
using namespace WireCell::PointCloud::Tree;

namespace {

    using array_type = boost::multi_array<float, 2>;

    struct GoldenArray {
        std::string name;
        array_type array;

        GoldenArray(const std::string& name, size_t nrows, size_t ncols)
            : name(name)
            , array(boost::extents[nrows][ncols])
        {
        }
    };

    using GoldenArrays = std::vector<GoldenArray>;

    Dataset make_ctpc(double offset)
    {
        return Dataset({
            {"x", Array({offset + 1.25, offset - 2.5})},
            {"y", Array({offset + 3.5, offset - 4.75})},
            {"charge", Array({offset + 5.125, offset + 6.25})},
            {"charge_err", Array({offset + 0.5, offset + 0.75})},
            {"cident", Array({10, 11})},
            {"wind", Array({20, 21})},
            {"slice_index", Array({30, 31})},
        });
    }

    Dataset make_blob_scalar(double charge, double center_x, int npoints, int wire_offset, int slice_index)
    {
        using fa_float_t = WireCell::PointCloud::Facade::float_t;
        using fa_int_t = WireCell::PointCloud::Facade::int_t;
        return Dataset({
            {"charge", Array({fa_float_t(charge)})},
            {"center_x", Array({fa_float_t(center_x)})},
            {"center_y", Array({fa_float_t(0.0)})},
            {"center_z", Array({fa_float_t(0.0)})},
            {"face", Array({fa_int_t(1)})},
            {"npoints", Array({fa_int_t(npoints)})},
            {"slice_index_min", Array({fa_int_t(slice_index)})},
            {"slice_index_max", Array({fa_int_t(slice_index + 1)})},
            {"u_wire_index_min", Array({fa_int_t(wire_offset)})},
            {"u_wire_index_max", Array({fa_int_t(wire_offset + 10)})},
            {"v_wire_index_min", Array({fa_int_t(wire_offset + 100)})},
            {"v_wire_index_max", Array({fa_int_t(wire_offset + 110)})},
            {"w_wire_index_min", Array({fa_int_t(wire_offset + 200)})},
            {"w_wire_index_max", Array({fa_int_t(wire_offset + 210)})},
            {"max_wire_interval", Array({fa_int_t(10)})},
            {"min_wire_interval", Array({fa_int_t(10)})},
            {"max_wire_type", Array({fa_int_t(0)})},
            {"min_wire_type", Array({fa_int_t(2)})},
        });
    }

    Dataset make_corners(double xbeg, double xend)
    {
        using fa_float_t = WireCell::PointCloud::Facade::float_t;
        return Dataset({
            {"x", Array({fa_float_t(xbeg), fa_float_t(xbeg), fa_float_t(xend), fa_float_t(xend)})},
            {"y", Array({fa_float_t(-0.5), fa_float_t(0.5), fa_float_t(0.5), fa_float_t(-0.5)})},
            {"z", Array({fa_float_t(-0.25), fa_float_t(-0.25), fa_float_t(0.25), fa_float_t(0.25)})},
        });
    }

    Points::node_ptr make_cluster(double offset)
    {
        using WireCell::PointTesting::make_janky_track;
        auto cluster = std::make_unique<Points::node_t>();
        for (int bind = 0; bind < 2; ++bind) {
            const double xbeg = offset + bind * 1.25;
            const double xend = xbeg + 1.0;
            auto track = make_janky_track(Ray(Point(xbeg, 0, 0), Point(xend, 0, 0)));
            const int npoints = track.size_major();
            cluster->insert(Points({
                {"scalar", make_blob_scalar(10.0 + offset + bind, 0.5 * (xbeg + xend), npoints,
                                            int(offset * 10) + bind * 10, bind * 4)},
                {"corner", make_corners(xbeg, xend)},
                {"3d", std::move(track)},
            }));
        }
        return cluster;
    }

    Points::node_ptr make_post_grouping_fixture()
    {
        auto root = std::make_unique<Points::node_t>(Points({
            {"ctpc_f1p0", make_ctpc(100.0)},
            {"ctpc_f1p1", make_ctpc(200.0)},
            {"ctpc_f1p2", make_ctpc(300.0)},
        }));
        root->insert(make_cluster(0.0));
        root->insert(make_cluster(4.0));
        return root;
    }

    // Frozen verbatim from grouping2file() at 01dc76b0, except that each
    // completed array is retained in memory instead of being streamed to NPZ.
    // This is deliberately test-only and independent of make_reco_arrays().
    GoldenArrays legacy_01dc76b0_arrays(const Grouping& grouping)
    {
        GoldenArrays ret;

        const auto& gpcs = grouping.node()->value.local_pcs();
        std::regex ctpc_name_pattern("^ctpc_f([0-9]+)p([0-9]+)$");
        for (const auto& [name, pc] : gpcs) {
            std::smatch matches;
            if (!std::regex_match(name, matches, ctpc_name_pattern)) {
                continue;
            }

            size_t npts = pc.size_major();
            ret.emplace_back(name, npts, 7);
            auto& actpc = ret.back().array;
            const auto& x = pc.get("x")->elements<double>();
            const auto& y = pc.get("y")->elements<double>();
            const auto& charge = pc.get("charge")->elements<double>();
            const auto& charge_err = pc.get("charge_err")->elements<double>();
            const auto& cident = pc.get("cident")->elements<int>();
            const auto& wind = pc.get("wind")->elements<int>();
            const auto& slice_index = pc.get("slice_index")->elements<int>();
            for (size_t i = 0; i < npts; ++i) {
                actpc[i][0] = x[i];
                actpc[i][1] = y[i];
                actpc[i][2] = charge[i];
                actpc[i][3] = charge_err[i];
                actpc[i][4] = cident[i];
                actpc[i][5] = wind[i];
                actpc[i][6] = slice_index[i];
            }
        }
        size_t nblobs = 0;
        for (const auto& cluster : grouping.children()) {
            nblobs += cluster->children().size();
        }
        ret.emplace_back("blobs", nblobs, 38);
        auto& ablobs = ret.back().array;
        int gbidx = 0;
        std::unordered_map<const Blob*, int> b2idx;
        for (const auto& cluster : grouping.children()) {
            for (const auto& blob : cluster->children()) {
                ablobs[gbidx][0] = blob->charge();
                int ncorners = std::min(12, int(blob->corners().size()));
                ablobs[gbidx][1] = ncorners;
                for (int i = 0; i < ncorners; ++i) {
                    ablobs[gbidx][2 + i * 3] = blob->corners().at(i).x();
                    ablobs[gbidx][3 + i * 3] = blob->corners().at(i).y();
                    ablobs[gbidx][4 + i * 3] = blob->corners().at(i).z();
                }
                b2idx[blob] = gbidx;
                gbidx++;
            }
        }

        int npoints = 0;
        int nedges = 0;
        for (const auto& cluster : grouping.children()) {
            cluster->Create_graph(true);
            const MCUGraph& g = *(cluster->get_graph());
            npoints += boost::num_vertices(g);
            nedges += boost::num_edges(g);
        }

        ret.emplace_back("points", npoints, 6);
        ret.emplace_back("ppedges", nedges, 3);
        auto& apoints = ret[ret.size() - 2].array;
        auto& aedges = ret[ret.size() - 1].array;

        int gcidx = 0;
        int gpidx = 0;
        int geidx = 0;
        int gpoffset = 0;
        for (const auto& cluster : grouping.children()) {
            const MCUGraph& g = *(cluster->get_graph());
            auto vrange = boost::vertices(g);
            for (auto vit = vrange.first; vit != vrange.second; ++vit) {
                auto v = *vit;
                gpidx = gpoffset + v;
                if (gpidx >= npoints) {
                    raise<ValueError>("graph2json: gpidx %d >= npoints %d", gpidx, npoints);
                }
                apoints[gpidx][0] = cluster->point3d(g[v].index).x();
                apoints[gpidx][1] = cluster->point3d(g[v].index).y();
                apoints[gpidx][2] = cluster->point3d(g[v].index).z();
                const auto [tmppt, blob] = cluster->get_closest_point_blob(
                    {apoints[gpidx][0], apoints[gpidx][1], apoints[gpidx][2]});
                apoints[gpidx][3] = blob->charge() / blob->npoints();
                apoints[gpidx][4] = b2idx[blob];
                apoints[gpidx][5] = gcidx;
            }
            auto erange = boost::edges(g);
            for (auto eit = erange.first; eit != erange.second; ++eit) {
                auto e = *eit;
                auto source = boost::source(e, g);
                auto target = boost::target(e, g);
                aedges[geidx][0] = source + gpoffset;
                aedges[geidx][1] = target + gpoffset;
                aedges[geidx][2] = boost::get(boost::edge_weight, g, e);
                geidx++;
            }
            gpoffset += boost::num_vertices(g);
            gcidx += 1;
        }
        return ret;
    }

}

TEST_CASE("make_reco_arrays preserves the reconstruction-array contract")
{
    Points::node_t root(Points({
        {"ctpc_f1p2", make_ctpc(200.0)},
        {"not_a_ctpc", make_ctpc(300.0)},
        {"ctpc_f1p0", make_ctpc(100.0)},
    }));
    auto* grouping = root.value.facade<Grouping>();
    REQUIRE(grouping);

    const auto arrays = make_reco_arrays(*grouping);
    REQUIRE(arrays.arrays.size() == 5);

    CHECK(arrays.arrays[0].name == "ctpc_f1p0");
    CHECK(arrays.arrays[1].name == "ctpc_f1p2");
    CHECK(arrays.arrays[2].name == "blobs");
    CHECK(arrays.arrays[3].name == "points");
    CHECK(arrays.arrays[4].name == "ppedges");

    const auto& ctpc = *arrays.arrays[0].array;
    CHECK(ctpc.shape()[0] == 2);
    CHECK(ctpc.shape()[1] == 7);
    CHECK(ctpc[0][0] == float(101.25));
    CHECK(ctpc[1][0] == float(97.5));
    CHECK(ctpc[0][1] == float(103.5));
    CHECK(ctpc[0][2] == float(105.125));
    CHECK(ctpc[0][3] == float(100.5));
    CHECK(ctpc[0][4] == float(10));
    CHECK(ctpc[0][5] == float(20));
    CHECK(ctpc[0][6] == float(30));

    CHECK(arrays.arrays[2].array->shape()[0] == 0);
    CHECK(arrays.arrays[2].array->shape()[1] == 38);
    CHECK(arrays.arrays[3].array->shape()[0] == 0);
    CHECK(arrays.arrays[3].array->shape()[1] == 6);
    CHECK(arrays.arrays[4].array->shape()[0] == 0);
    CHECK(arrays.arrays[4].array->shape()[1] == 3);

    RecoArrays initialized;
    const auto& blank_blobs = initialized.append("blobs", 2, 38);
    for (size_t row = 0; row < blank_blobs.shape()[0]; ++row) {
        for (size_t col = 0; col < blank_blobs.shape()[1]; ++col) {
            CHECK(blank_blobs[row][col] == 0.0f);
        }
    }

    const std::string filename = "doctest_recoarrays.npz";
    grouping2file(*grouping, filename);

    boost::iostreams::filtering_istream input;
    Stream::input_filters(input, filename);
    REQUIRE_FALSE(input.empty());
    for (const auto& expected : arrays.arrays) {
        std::string name;
        size_t file_size = 0;
        custard::read(input, name, file_size);
        pigenc::File serialized;
        serialized.read(input);
        REQUIRE(input);
        CHECK(name == expected.name);
        const auto& header = serialized.header();
        REQUIRE(header.shape().size() == 2);
        CHECK(header.shape()[0] == expected.array->shape()[0]);
        CHECK(header.shape()[1] == expected.array->shape()[1]);
        CHECK(header.dtype() == "f4");
        CHECK_FALSE(header.fortran_order());
        CHECK(file_size == header.file_size());
        const auto bytes = serialized.data();
        REQUIRE(bytes.size() == expected.array->num_elements() * sizeof(float));
        if (!bytes.empty()) {
            CHECK(std::memcmp(expected.array->data(), bytes.data(), bytes.size()) == 0);
        }
    }
    input.pop();
    CHECK(std::remove(filename.c_str()) == 0);
}

TEST_CASE("make_reco_arrays exactly matches independent 01dc76b0 array construction")
{
    using namespace WireCell::Aux::TensorDM;

    auto frozen = make_post_grouping_fixture();
    const auto tensors = as_tensors(*frozen, "golden/post-grouping");

    // Recreate independent trees so Create_graph() caches can not leak from
    // one implementation into the other or make call order affect the test.
    auto legacy_tree = as_pctree(tensors, "golden/post-grouping");
    auto extracted_tree = as_pctree(tensors, "golden/post-grouping");
    REQUIRE(legacy_tree);
    REQUIRE(extracted_tree);

    auto* legacy_grouping = legacy_tree->value.facade<Grouping>();
    auto* extracted_grouping = extracted_tree->value.facade<Grouping>();
    REQUIRE(legacy_grouping);
    REQUIRE(extracted_grouping);

    const auto expected = legacy_01dc76b0_arrays(*legacy_grouping);
    const auto actual = make_reco_arrays(*extracted_grouping);
    REQUIRE(expected.size() == 6);
    REQUIRE(actual.arrays.size() == expected.size());

    for (size_t ind = 0; ind < expected.size(); ++ind) {
        const auto& lhs = expected[ind];
        const auto& rhs = actual.arrays[ind];
        CHECK(rhs.name == lhs.name);
        REQUIRE(rhs.array->shape()[0] == lhs.array.shape()[0]);
        REQUIRE(rhs.array->shape()[1] == lhs.array.shape()[1]);
        REQUIRE(rhs.array->num_elements() == lhs.array.num_elements());
        CHECK(std::memcmp(rhs.array->data(), lhs.array.data(), lhs.array.num_elements() * sizeof(float)) == 0);
    }

    CHECK(actual.arrays[3].array->shape()[0] == 4);
    CHECK(actual.arrays[4].array->shape()[0] > 0);
    CHECK(actual.arrays[5].array->shape()[0] > 0);
}
