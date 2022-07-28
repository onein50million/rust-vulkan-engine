glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv
glslc shader.comp -o comp.spv

glslc ui.vert -o ui_vert.spv
glslc ui.frag -o ui_frag.spv

glslc province/shader.vert -o province/vert.spv
glslc province/shader.frag -o province/frag.spv

glslc cubemap/shader.vert -o cubemap/vert.spv
glslc cubemap/shader.frag -o cubemap/frag.spv

glslc planet/normal.comp -o planet/normal.spv

glslc irradiance.comp -o irradiance.spv
glslc environment.comp -o environment.spv

pause