set(TARGET llama-onnx-export-dump)
add_executable(${TARGET} onnx-export-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama mtmd ${CMAKE_THREAD_LIBS_INIT})
# We poke at ggml_cgraph internals (n_leafs, leafs, n_nodes, nodes) which live
# in the impl header; export-graph-ops does the same. We also include the
# mtmd directory so we can call into clip.h for vision-encoder graph dumping.
target_include_directories(${TARGET} PRIVATE ../../common ../../tools ${PROJECT_SOURCE_DIR}/ggml/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
