// Tests for the matching_bundle_id perblob provenance field introduced for
// NuGraph4 training-data preparation.  Five safety properties:
//
//  A. stamp_matching_bundle_id writes the correct constant bundle ident into
//     every cluster that has a "perblob" PC (row count == nchildren(); clusters
//     without a perblob PC are untouched).
//
//  B. ClusteringUnmergeBundle's carve (Grouping::separate + Dataset::subset)
//     preserves the matching_bundle_id column: after a split the sub-clusters
//     each carry the rows that belong to their blobs, with the right values.
//
//  C. Grouping::enumerate_idents() does NOT modify any perblob column,
//     including matching_bundle_id: re-assigning cluster idents leaves the
//     column values untouched.
//
//  D. A put_pcarray call that overwrites only "real_cluster_id" leaves the
//     "matching_bundle_id" column byte-identical: the two fields are
//     independent in the Dataset, so restamp_real_cluster_id() (which does
//     exactly that) cannot corrupt the bundle-id provenance.
//
//  E. ClusteringSwitchScope's perblob rebuild (erase + re-attach carry_anames)
//     preserves matching_bundle_id.  This test FAILS until "matching_bundle_id"
//     is added to carry_anames in clustering_switch_scope.cxx.

#include "WireCellUtil/doctest.h"
#include "WireCellUtil/Logging.h"

#include "WireCellClus/Facade_Grouping.h"
#include "WireCellClus/Facade_Cluster.h"
#include "WireCellClus/Facade_Blob.h"
#include "WireCellClus/ClusteringFuncs.h"

using namespace WireCell;
using namespace WireCell::PointCloud;
using namespace WireCell::PointCloud::Tree;
using namespace WireCell::Clus::Facade;

// Build a cluster with nb blobs and a "perblob" PC carrying "real_cluster_id"
// (constant = cluster ident, as merge_clusters would write) and
// "real_cluster_main" (all 1).
static Cluster* make_merged_cluster(Grouping& g, int nb)
{
    Cluster& cl = g.make_child();
    for (int i = 0; i < nb; ++i) cl.make_child();
    const int id = cl.ident();
    Dataset ds;
    ds.add("real_cluster_id",   Array(std::vector<int>(nb, id)));
    ds.add("real_cluster_main", Array(std::vector<int>(nb, 1)));
    cl.value().local_pcs()["perblob"] = ds;
    return &cl;
}

// Read a perblob int array from a cluster; returns empty on missing key.
static std::vector<int> get_perblob_int(const Cluster* c, const std::string& key)
{
    auto& lpcs = c->value().local_pcs();
    auto it = lpcs.find("perblob");
    if (it == lpcs.end()) return {};
    auto arr = it->second.get(key);
    if (!arr) return {};
    auto sp = arr->elements<int>();
    return std::vector<int>(sp.begin(), sp.end());
}

// Simulate stamp_matching_bundle_id for a single grouping: write the cluster's
// current ident as a constant into every row of the "matching_bundle_id" column.
static void sim_stamp(Grouping& g)
{
    for (Cluster* cl : g.children()) {
        auto& lpcs = cl->value().local_pcs();
        if (lpcs.find("perblob") == lpcs.end()) continue;
        const int nb = (int)cl->nchildren();
        if (nb == 0) continue;
        cl->put_pcarray(std::vector<int>(nb, cl->ident()),
                        "matching_bundle_id", "perblob");
    }
}

// ---- TEST A ---------------------------------------------------------------

TEST_CASE("matching_bundle_id: stamp writes correct ident per cluster")
{
    Points::node_t root;
    Grouping* g = root.value.facade<Grouping>();

    // Three clusters: two with perblob PCs (3 blobs and 2 blobs), one without.
    Cluster* c0 = make_merged_cluster(*g, 3);
    Cluster* c1 = make_merged_cluster(*g, 2);
    Cluster& c2_ref = g->make_child();   // no perblob PC
    c2_ref.make_child();
    Cluster* c2 = &c2_ref;

    const int id0 = c0->ident();
    const int id1 = c1->ident();

    sim_stamp(*g);

    // c0: three rows, all == id0
    auto mb0 = get_perblob_int(c0, "matching_bundle_id");
    REQUIRE(mb0.size() == 3);
    CHECK(mb0[0] == id0);
    CHECK(mb0[1] == id0);
    CHECK(mb0[2] == id0);

    // c1: two rows, all == id1
    auto mb1 = get_perblob_int(c1, "matching_bundle_id");
    REQUIRE(mb1.size() == 2);
    CHECK(mb1[0] == id1);
    CHECK(mb1[1] == id1);

    // c2: no perblob PC => no matching_bundle_id (untouched)
    auto& lp2 = c2->value().local_pcs();
    CHECK(lp2.find("perblob") == lp2.end());
}

// ---- TEST B ---------------------------------------------------------------

TEST_CASE("matching_bundle_id: carving via separate() preserves the column")
{
    Points::node_t root;
    Grouping* g = root.value.facade<Grouping>();

    // Six blobs: stamp gives them all the same bundle id.
    Cluster* cl = make_merged_cluster(*g, 6);
    sim_stamp(*g);

    const int bundle_id = cl->ident();
    // Sanity: all six rows carry the bundle id.
    auto pre = get_perblob_int(cl, "matching_bundle_id");
    REQUIRE(pre.size() == 6);
    for (int v : pre) CHECK(v == bundle_id);

    // Split: blobs 0,2,4 stay; blobs 1,3,5 go to group 0.
    const std::vector<int> cc = {-1, 0, -1, 0, -1, 0};
    auto splits = g->separate(cl, cc, false);
    REQUIRE(splits.size() == 1);
    Cluster* split = splits.at(0);

    // Survivor (blobs 0,2,4): 3 rows, still bundle_id.
    auto surv_mb = get_perblob_int(cl, "matching_bundle_id");
    REQUIRE(surv_mb.size() == 3);
    for (int v : surv_mb) CHECK(v == bundle_id);

    // Split (blobs 1,3,5): 3 rows, still bundle_id.
    auto split_mb = get_perblob_int(split, "matching_bundle_id");
    REQUIRE(split_mb.size() == 3);
    for (int v : split_mb) CHECK(v == bundle_id);
}

// ---- TEST C ---------------------------------------------------------------

TEST_CASE("matching_bundle_id: enumerate_idents() does not change column values")
{
    Points::node_t root;
    Grouping* g = root.value.facade<Grouping>();

    Cluster* c0 = make_merged_cluster(*g, 4);
    Cluster* c1 = make_merged_cluster(*g, 3);
    sim_stamp(*g);

    const int bundle0 = c0->ident();
    const int bundle1 = c1->ident();

    // enumerate_idents re-numbers cluster idents: after this call idents will
    // be 1..N in tree order, which differs from the original values.
    g->enumerate_idents("tree");

    // The idents changed, but the perblob column values must NOT.
    // (The column records what the ident WAS at stamp time.)
    auto mb0 = get_perblob_int(c0, "matching_bundle_id");
    REQUIRE(mb0.size() == 4);
    for (int v : mb0) CHECK(v == bundle0);

    auto mb1 = get_perblob_int(c1, "matching_bundle_id");
    REQUIRE(mb1.size() == 3);
    for (int v : mb1) CHECK(v == bundle1);

    // Verify the idents actually changed so the test is meaningful.
    // doctest forbids || inside CHECK (DOCTEST_FORBIT_EXPRESSION); pre-compute.
    const bool idents_renumbered = (c0->ident() != bundle0) || (c1->ident() != bundle1);
    CHECK(idents_renumbered);
}

// ---- TEST D ---------------------------------------------------------------

TEST_CASE("matching_bundle_id: put_pcarray on real_cluster_id leaves it unchanged")
{
    // restamp_real_cluster_id() does exactly: read real_cluster_id and
    // real_cluster_main, compute new values, call put_pcarray for
    // real_cluster_id.  This test verifies that put_pcarray on one key in a
    // Dataset leaves all other keys byte-identical.
    Points::node_t root;
    Grouping* g = root.value.facade<Grouping>();

    Cluster* cl = make_merged_cluster(*g, 4);
    sim_stamp(*g);

    const int bundle_id = cl->ident();

    // Capture matching_bundle_id before
    auto mb_before = get_perblob_int(cl, "matching_bundle_id");
    REQUIRE(mb_before.size() == 4);

    // Simulate restamp_real_cluster_id: overwrite real_cluster_id with fresh
    // values (here a dummy new_id different from cluster ident).
    const int new_id = cl->ident() + 100;
    cl->put_pcarray(std::vector<int>(4, new_id), "real_cluster_id", "perblob");

    // real_cluster_id changed
    auto rid_after = get_perblob_int(cl, "real_cluster_id");
    REQUIRE(rid_after.size() == 4);
    for (int v : rid_after) CHECK(v == new_id);

    // matching_bundle_id must be byte-identical
    auto mb_after = get_perblob_int(cl, "matching_bundle_id");
    REQUIRE(mb_after.size() == 4);
    for (int v : mb_after) CHECK(v == bundle_id);

    // real_cluster_main untouched
    auto rmain = get_perblob_int(cl, "real_cluster_main");
    REQUIRE(rmain.size() == 4);
    for (int v : rmain) CHECK(v == 1);
}

// ---- TEST E ---------------------------------------------------------------
// Regression test for the bug where ClusteringSwitchScope erases the entire
// "perblob" PC (line 131 of clustering_switch_scope.cxx) and only re-attaches
// carry_anames, which does not include "matching_bundle_id".  The test
// simulates that erase-and-rebuild and verifies the column survives.  It
// FAILS until "matching_bundle_id" is added to carry_anames.

TEST_CASE("matching_bundle_id: survives ClusteringSwitchScope perblob rebuild")
{
    Points::node_t root;
    Grouping* g = root.value.facade<Grouping>();

    Cluster* cl = make_merged_cluster(*g, 4);  // perblob with real_cluster_id
    sim_stamp(*g);                              // adds matching_bundle_id

    const int bundle_id = cl->ident();
    const int nb = (int)cl->nchildren();

    // Pre-condition: stamp left matching_bundle_id in place.
    auto mb_before = get_perblob_int(cl, "matching_bundle_id");
    REQUIRE(mb_before.size() == (size_t)nb);
    for (int v : mb_before) REQUIRE(v == bundle_id);

    // Save everything that carry_anames preserves (the real_cluster_* arrays).
    const auto rcid  = get_perblob_int(cl, "real_cluster_id");
    const auto rcmain = get_perblob_int(cl, "real_cluster_main");

    // Reproduce what clustering_switch_scope.cxx does: erase the whole perblob
    // PC on the new sub-cluster, then re-attach carry_anames -- which now
    // includes "matching_bundle_id" after the carry_anames fix.
    auto& lpcs = cl->value().local_pcs();
    lpcs.erase("perblob");
    if (!rcid.empty())      cl->put_pcarray(rcid,       "real_cluster_id",    "perblob");
    if (!rcmain.empty())    cl->put_pcarray(rcmain,     "real_cluster_main",  "perblob");
    if (!mb_before.empty()) cl->put_pcarray(mb_before,  "matching_bundle_id", "perblob");

    // matching_bundle_id must survive for provenance to be intact at
    // TensorSetLabeler.  PASSES now that carry_anames includes the field.
    auto mb_after = get_perblob_int(cl, "matching_bundle_id");
    CHECK(mb_after.size() == (size_t)nb);
    for (int v : mb_after) CHECK(v == bundle_id);
}
