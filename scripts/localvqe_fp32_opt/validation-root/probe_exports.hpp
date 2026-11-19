#include <cstdint>
#include <stdexcept>

// Append-only diagnostic ABI. Called only by the separate state runner, never
// from the timed VCP callback. Capture logical F32 tensor elements, not freed or
// buffer-reused intermediates. dbg_taps are explicitly retained graph outputs.
namespace lvq_probe {
struct sink {
    FILE* data; FILE* meta; std::size_t offset=0;
    void value(const std::string& name,const void* p,std::size_t count,const char* type) {
        const std::size_t bytes=count*(std::strcmp(type,"f32")==0?4:8);
        if(meta)std::fprintf(meta,"%s,%s,%zu,%zu\n",name.c_str(),type,count,offset);
        if(bytes && std::fwrite(p,1,bytes,data)!=bytes)throw std::runtime_error("probe write failed");
        offset+=bytes;
    }
    void vec(const std::string& name,const std::vector<float>& x){value(name,x.data(),x.size(),"f32");}
    void scalar(const std::string& name,std::int64_t x){value(name,&x,1,"i64");}
    void tensor(const std::string& name,struct ggml_tensor* t){
        if(!t)return;
        if(t->type!=GGML_TYPE_F32)throw std::runtime_error("probe expects F32 state");
        const auto count=ggml_nelements(t);std::vector<float> result(count);
        if(ggml_is_contiguous(t))ggml_backend_tensor_get(t,result.data(),0,count*sizeof(float));
        else {
            std::vector<unsigned char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
            for(std::int64_t z=0;z<count;++z){auto left=z;std::size_t at=0;for(int k=0;k<4;++k){at+=(left%t->ne[k])*t->nb[k];left/=t->ne[k];}std::memcpy(&result[z],raw.data()+at,4);}
        }
        value(name,result.data(),result.size(),"f32");
    }
};
}

extern "C" __attribute__((visibility("default"))) int itu_lvq_probe_dump(localvqe_ctx_t h,FILE* data,FILE* meta){
    try {
        if(!h||!data)return -1;
        auto& c=*reinterpret_cast<localvqe_ctx*>(h);auto& g=c.stream_graph;auto& d=c.daf;
        lvq_probe::sink s{data,meta};if(meta)std::fprintf(meta,"name,dtype,count,byte_offset\n");
        for(std::size_t i=0;i<g.conv_hist_in.size();++i)s.tensor("conv_hist_in_"+std::to_string(i),g.conv_hist_in[i]);
        // conv_hist_out may alias history after the mandated two-phase copy;
        // capture the persistent in-state above, not overwritten old outputs.
        for(std::size_t i=0;i<g.conv_win.size();++i)s.tensor("conv_window_"+std::to_string(i),g.conv_win[i]);
        s.tensor("s4d_real",g.s4d_h_real_in);s.tensor("s4d_imag",g.s4d_h_imag_in);
        s.tensor("align_K",g.align_K_hist_in);s.tensor("align_ref",g.align_ref_hist_in);s.tensor("align_smooth",g.align_smooth_hist_in);s.tensor("ccm_history",g.ccm_hist_in);
        // Debug taps are ggml_set_output'd; their values survive graph compute.
        for(std::size_t i=0;i<g.dbg_taps.size();++i)s.tensor("tap_"+g.dbg_names[i],g.dbg_taps[i]);
        s.vec("pcm_mic",c.pcm_hist_mic);s.vec("pcm_ref",c.pcm_hist_ref);s.vec("ola",c.ola);s.vec("enh_window",c.enh_window);
        s.vec("daf_e",c.daf_e);s.vec("daf_yhat",c.daf_yhat);
        s.vec("daf_Hr",d.Hr);s.vec("daf_Hi",d.Hi);s.vec("daf_Xr",d.Xr_r);s.vec("daf_Xi",d.Xr_i);s.vec("daf_P",d.P);s.vec("daf_x_old",d.x_old);
        s.vec("daf_global_gru",d.hg);s.vec("daf_bin_gru",d.hb);s.vec("daf_partition_gru",d.hp);
        s.vec("gcc_Sr",d.gcc_Sr);s.vec("gcc_Si",d.gcc_Si);s.vec("gcc_mic_ring",d.mic_ring);s.vec("gcc_ref_ring",d.ref_ring);s.vec("reference_delay",d.ref_dline);
        const float scalars[]={d.gcc_maxrms,d.gcc_conf};s.value("gcc_floats",scalars,2,"f32");
        s.scalar("partition_index",d.p_idx);s.scalar("samples_seen",d.n_seen);s.scalar("reference_delay_position",d.ref_dpos);s.scalar("gcc_shift",d.cur_shift);s.scalar("gcc_locked",d.gcc_locked);s.scalar("prealign",d.enable_prealign);
        return static_cast<int>(s.offset);
    }catch(...){return -2;}
}
