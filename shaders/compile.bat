glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv
glslc shader.comp -o comp.spv

glslc postprocess_subpass/shader.vert -o postprocess_subpass/vert.spv
glslc postprocess_subpass/shader.frag -o postprocess_subpass/frag.spv

glslc 3dSDF.comp -o 3dSDF.spv
glslc bakeVoronoi.comp -o bakeVoronoi.spv

glslc ui.vert -o ui_vert.spv
glslc ui.frag -o ui_frag.spv

glslc cubemap/shader.vert -o cubemap/vert.spv
glslc cubemap/shader.frag -o cubemap/frag.spv

glslc irradiance.comp -o irradiance.spv
glslc environment.comp -o environment.spv

pause