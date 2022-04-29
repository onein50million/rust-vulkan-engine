glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv
glslc shader.comp -o comp.spv

glslc ui.vert -o ui_vert.spv
glslc ui.frag -o ui_frag.spv

glslc line/shader.vert -o line/vert.spv
glslc line/shader.frag -o line/frag.spv

glslc irradiance.comp -o irradiance.spv
glslc environment.comp -o environment.spv

pause