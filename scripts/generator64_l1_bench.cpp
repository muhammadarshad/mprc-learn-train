// Exhaustive warm-L1 traversal benchmark over the 64 x 113 byte cache tile.
//
// This measures PHYSICAL traversal only. It does not decide MPRC structural quality.
// All full-period additive generators g (odd 1..63) visit the same 64 rows exactly
// once per pass. The tile is 7232 bytes = 113 x 64-byte cache lines.
//
// Build: c++ -O3 -march=native -std=c++20 generator64_l1_bench.cpp -o genbench
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

static constexpr int H=64;
static constexpr int W=113;
static constexpr int GCOUNT=32;
static constexpr int ROUNDS=9;
static constexpr int PASSES=5000;

static volatile std::uint64_t sink_value=0;

static inline std::uint64_t run_passes(volatile std::uint8_t* tile, int g, int passes) {
    std::uint64_t s=0;
    for (int rep=0; rep<passes; ++rep) {
        int q=0;
        for (int t=0; t<H; ++t) {
            volatile std::uint8_t* row=tile + (std::size_t)q*W;
            for (int w=0; w<W; ++w) s += row[w];
            q=(q+g)&63;
        }
    }
    sink_value ^= s;
    return s;
}

int main() {
    std::vector<std::uint8_t> storage(H*W + 64);
    std::uintptr_t base=(reinterpret_cast<std::uintptr_t>(storage.data()) + 63u) & ~std::uintptr_t(63u);
    volatile std::uint8_t* tile=reinterpret_cast<volatile std::uint8_t*>(base);
    for (int i=0;i<H*W;++i) const_cast<std::uint8_t*>(tile)[i]=(std::uint8_t)((i*37+11)&255);

    std::array<int,GCOUNT> gs{};
    int gi=0;
    for(int g=1;g<64;g+=2) gs[gi++]=g;

    std::array<std::array<double,ROUNDS>,GCOUNT> times{};

    // Warm the complete tile.
    run_passes(tile,7,200);

    for(int r=0;r<ROUNDS;++r) {
        // Deterministic rotation/reversal of generator order to reduce temporal bias.
        std::array<int,GCOUNT> order=gs;
        std::rotate(order.begin(),order.begin()+((r*7)%GCOUNT),order.end());
        if(r&1) std::reverse(order.begin(),order.end());

        for(int g:order) {
            auto t0=std::chrono::steady_clock::now();
            run_passes(tile,g,PASSES);
            auto t1=std::chrono::steady_clock::now();
            double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/PASSES;
            int idx=(g-1)/2;
            times[idx][r]=ns;
        }
    }

    std::ofstream out("results/generator64_l1_bench.csv");
    out<<"g,median_ns_per_full_64x113_pass,min_ns,max_ns,bytes_per_pass\n";

    struct Row {int g; double med,mi,ma;};
    std::vector<Row> rows;
    for(int idx=0;idx<GCOUNT;++idx) {
        auto a=times[idx];
        std::sort(a.begin(),a.end());
        Row row{gs[idx],a[ROUNDS/2],a.front(),a.back()};
        rows.push_back(row);
        out<<row.g<<","<<row.med<<","<<row.mi<<","<<row.ma<<","<<(H*W)<<"\n";
    }
    out.close();

    std::sort(rows.begin(),rows.end(),[](const Row&a,const Row&b){
        if(a.med!=b.med) return a.med<b.med;
        return a.g<b.g;
    });

    std::ofstream js("results/generator64_l1_bench.json");
    js<<"{\n";
    js<<"  \"tile_bytes\": "<<(H*W)<<",\n";
    js<<"  \"cacheline_bytes\": 64,\n";
    js<<"  \"tile_cache_lines\": "<<((H*W)/64)<<",\n";
    js<<"  \"passes_per_measurement\": "<<PASSES<<",\n";
    js<<"  \"rounds\": "<<ROUNDS<<",\n";
    js<<"  \"fastest_by_median\": "<<rows.front().g<<",\n";
    js<<"  \"fastest_median_ns\": "<<rows.front().med<<",\n";
    double g7=0;
    for(auto&r:rows) if(r.g==7) g7=r.med;
    js<<"  \"g7_median_ns\": "<<g7<<",\n";
    js<<"  \"g7_slowdown_vs_fastest_pct\": "<<((g7/rows.front().med-1.0)*100.0)<<",\n";
    js<<"  \"claim_boundary\": \"Warm-L1 microbenchmark on one GitHub Actions CPU; performance is hardware/noise dependent and is not a structural theorem.\"\n";
    js<<"}\n";
    js.close();

    std::printf("fastest g=%d median %.3f ns/pass; g7 %.3f ns/pass; sink=%llu\n",
                rows.front().g,rows.front().med,g7,(unsigned long long)sink_value);
    return 0;
}
