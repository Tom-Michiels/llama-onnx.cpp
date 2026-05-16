set(TARGET llama-onnx-export-dump)
add_executable(${TARGET} onnx-export-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# We poke at ggml_cgraph internals (n_leafs, leafs, n_nodes, nodes) which live
# in the impl header; export-graph-ops does the same.
target_include_directories(${TARGET} PRIVATE ../../common ${PROJECT_SOURCE_DIR}/ggml/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
