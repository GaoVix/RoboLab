#include "NRD.h"
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <atomic>

static_assert(sizeof(nrd::CommonSettings)==380, "NRD 3.2.3 layout mismatch");
static_assert(offsetof(nrd::CommonSettings,timeDeltaBetweenFrames)==0x158, "NRD time field mismatch");
static std::atomic<unsigned long> sequence{0};
static std::mutex call_mutex;
static std::mutex log_mutex;
static std::unordered_map<void*, unsigned> instance_ids;

static void log_line(const char* line, int n) {
    const char* path=getenv("ROBO_NRD_LOG");
    if(!path) return;
    std::lock_guard<std::mutex> lock(log_mutex);
    int fd=open(path,O_WRONLY|O_CREAT|O_APPEND,0600);
    if(fd>=0) { auto a=write(fd,line,n);auto b=write(fd,"\n",1);close(fd);if(a!=n || b!=1) _exit(102); }
}

// NRD's constructor imports libc rand(). Only calls originating in NRD 3.2.3
// receive the fixed seed; every other process/module receives its real value.
extern "C" int rand() noexcept {
    using Fn=int(*)();
    static Fn real=(Fn)dlsym(RTLD_NEXT,"rand");
    if(!real) _exit(96);
    int value=real();
    void* caller=__builtin_return_address(0); Dl_info info{};
    dladdr(caller,&info);
    if(info.dli_fname && strstr(info.dli_fname,"libNRD.so.3.2.3")) {
        char line[1400];
        int n=snprintf(line,sizeof(line),"{\"kind\":\"nrd_seed\",\"seq\":%lu,\"module\":\"%s\",\"caller_offset\":%zu,\"original\":%d,\"returned\":0}",sequence.fetch_add(1),info.dli_fname,(size_t)((char*)caller-(char*)info.dli_fbase),value);
        log_line(line,n);
        return 0;
    }
    return value;
}

static int find_library(dl_phdr_info* info,size_t,void* out) {
    if(info->dlpi_name && strstr(info->dlpi_name,"libNRD.so.3.2.3")) {
        *(void**)out=dlopen(info->dlpi_name,RTLD_LAZY|RTLD_NOLOAD);
        return *(void**)out ? 1 : 0;
    }
    return 0;
}

NRD_API nrd::Result NRD_CALL nrd::GetComputeDispatches(nrd::Denoiser& denoiser,
        const nrd::CommonSettings& incoming,const nrd::DispatchDesc*& dispatches,uint32_t& count) {
    using Fn=nrd::Result(*)(nrd::Denoiser&,const nrd::CommonSettings&,const nrd::DispatchDesc*&,uint32_t&);
    static Fn real=[](){
        void* handle=nullptr;dl_iterate_phdr(find_library,&handle);
        if(!handle) _exit(97);
        auto get_desc=(const nrd::LibraryDesc&(*)())dlsym(handle,"GetLibraryDesc");
        if(!get_desc) _exit(98);
        const auto& desc=get_desc();
        if(desc.versionMajor!=3 || desc.versionMinor!=2 || desc.versionBuild!=3) _exit(99);
        Fn fn=(Fn)dlsym(handle,"GetComputeDispatches");
        if(!fn || fn==&nrd::GetComputeDispatches) _exit(100);
        return fn;
    }();
    void* object=&denoiser;
    unsigned instance_id;
    {
        std::lock_guard<std::mutex> lock(call_mutex);
        auto it=instance_ids.find(object);
        if(it==instance_ids.end()) it=instance_ids.emplace(object,(unsigned)instance_ids.size()).first;
        instance_id=it->second;
    }
    // Do not hold a diagnostic lock while the actual NRD function runs.
    nrd::CommonSettings fixed=incoming;
    fixed.timeDeltaBetweenFrames=66.67f;
    // These two read-only offsets were verified in the installed binary's
    // UpdateCommonSettings implementation; no private object memory is written.
    uint32_t rng_before=0,rng_after=0;
    memcpy(&rng_before,(char*)object+0x260,4);
    auto result=real(denoiser,fixed,dispatches,count);
    memcpy(&rng_after,(char*)object+0x260,4);
    float copied_delta=0;
    memcpy(&copied_delta,(char*)object+0x2d0+0x158,4);
    if(copied_delta!=fixed.timeDeltaBetweenFrames) _exit(101);
    char line[1800];
    int n=snprintf(line,sizeof(line),"{\"kind\":\"nrd_dispatch\",\"seq\":%lu,\"instance\":%u,\"frameIndex\":%u,\"incoming_dt_ms\":%.9g,\"applied_dt_ms\":%.9g,\"readback_dt_ms\":%.9g,\"accumulation_mode\":%u,\"jitter\":[%.9g,%.9g],\"resolutionScale\":[%.9g,%.9g],\"rng_before\":%u,\"rng_after\":%u,\"dispatch_count\":%u,\"result\":%u}",sequence.fetch_add(1),instance_id,incoming.frameIndex,incoming.timeDeltaBetweenFrames,fixed.timeDeltaBetweenFrames,copied_delta,(unsigned)incoming.accumulationMode,incoming.cameraJitter[0],incoming.cameraJitter[1],incoming.resolutionScale[0],incoming.resolutionScale[1],rng_before,rng_after,count,(unsigned)result);
    log_line(line,n);
    return result;
}
