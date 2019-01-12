cmake_minimum_required(VERSION 3.8)

set (p31 PROJECT1)
project(p31)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++11 -pthread")
set(CMAKE_CXX_STANDARD 11)

include_directories(
        p31
        p31/src
)

add_executable(p31 p31/src/main.cc p31/src/input_image.cc p31/src/input_image.h p31/src/complex.cc p31/src/complex.h)

target_link_libraries (p31 ${CMAKE_THREAD_LIBS_INIT})


set (p32 PROJECT2)
project(p32)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Werror")

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

include_directories(
	p32
	p32/src
)

set(SOURCE_FILES
        
     p32/src/complex.h
       
     p32/src/complex.cc
       
     p32/src/input_image.h
        
     p32/src/input_image.cc
        
)


add_executable(p32 ${SOURCE_FILES} p32/main.cc)

target_link_libraries(p32 ${MPI_LIBRARIES})



if(MPI_COMPILE_FLAGS)
    
   set_target_properties(p32 PROPERTIES
            
       COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")

endif()



if(MPI_LINK_FLAGS)
    
   set_target_properties(p32 PROPERTIES
            
       LINK_FLAGS "${MPI_LINK_FLAGS}")

endif()


set (p33 PROJECT3)
project(p33)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

enable_language("CUDA")

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include_directories(
        p33
)

set(SOURCE_FILES
        
        )

ADD_EXECUTABLE(p33 p33/main.cu p33/input_image.h p33/input_image.cc p33/complex.h p33/complex.cc)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)


set (p34 PROJECT4)
project(p34)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread")
set(CMAKE_CXX_STANDARD 11)

include_directories(
        p34
        p34/src
)

add_executable(${PROJECT_NAME} p34/src/main.cc p34/src/input_image.cc p34/src/input_image.h p34/src/complex.cc p34/src/complex.h)
target_link_libraries (p34 ${CMAKE_THREAD_LIBS_INIT})
