from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GLU import *

from PyQt5.QtGui import QColor, QVector3D, QMatrix4x4
from PyQt5.QtCore import QRect

from shader import Shader
from textures import ReadHeightMap, ReadTexture, NumpyTexture

import numpy as np
class SensorData():
     
    vertexCount = 502
    terrainVertices = []
    terrainIndices = []


    def __init__(self, position):
        self.position = position
        self.setup()

    def draw(self, perspective , view, rewardMap):
        self.shader.use()
        self.shader.setMat4("perspective", perspective)
        self.shader.setMat4("view", view)

        glBindVertexArray(self.__vao)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, len(self.terrainIndices), GL_UNSIGNED_INT, None)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(0);
        self.colors = NumpyTexture(self.colors, rewardMap)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.colors)
        self.shader.setInt("terrainTexture", 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.heightMap)
        self.shader.setInt("heightMap", 1)
        self.shader.stop()
        # glActiveTexture(GL_TEXTURE1);
        # glBindTexture(GL_TEXTURE_2D, heightMap);
        # shader->setInt("heightMap", 1);

        # glActiveTexture(GL_TEXTURE2);
        # glBindTexture(GL_TEXTURE_2D, heightMap2);
        # shader->setInt("heightMap2", 2);

        
    def getVerticesCount(self, vertexCount):
        return vertexCount*vertexCount*3

    def getIndicesCount(self, vertexCount):
        return 6*(vertexCount-1)*(vertexCount-1)

    def setRewards(self, rewardMap):
        pass

    def getVertices(self, vertexCount):
        vertices = [0.0]*self.getVerticesCount(vertexCount)
        vertexPointer = 0
        for i in range(vertexCount):
            for j in range(vertexCount):
                vertices[vertexPointer*3] = (j/(vertexCount-1))*2.0 - 1.0
                vertices[vertexPointer*3+1] = 0
                vertices[vertexPointer*3+2] = (i/(vertexCount-1))*2.0 - 1.0
                vertexPointer = vertexPointer+1
        return vertices

    def getIndices(self, vertexCount):
        indices = [0.0]*self.getIndicesCount(vertexCount)
        pointer = 0
        for gz in range(vertexCount-1):
            for gx in range(vertexCount-1):
                topLeft = (gz*vertexCount)+gx
                topRight = topLeft + 1
                bottomLeft = ((gz+1)*vertexCount)+gx
                bottomRight = bottomLeft + 1
                indices[pointer] = topLeft
                pointer = pointer+1
                indices[pointer] = bottomLeft
                pointer = pointer+1
                indices[pointer] = topRight
                pointer = pointer+1
                indices[pointer] = topRight
                pointer = pointer+1
                indices[pointer] = bottomLeft
                pointer = pointer+1
                indices[pointer] = bottomRight
                pointer = pointer+1
        return indices



    def getObjectCoord(self, windowPos, perspective, view, viewport):
        modelView = QMatrix4x4()
        modelView*=view
        modelView*=self.model
        objectCoord = windowPos.unproject(modelView, perspective, self.np2QRect(viewport))
        return objectCoord

    def matrixTypeConversion(self, matrix):
        return QMatrix4x4(matrix.m11, matrix.m12,matrix.m13, matrix.m14,matrix.m21, matrix.m22,matrix.m23, matrix.m24,matrix.m31, matrix.m32,matrix.m33, matrix.m34,matrix.m41, matrix.m42,matrix.m43, matrix.m44)
    
    def np2QRect(self, raw_array):
        return QRect(raw_array[0], raw_array[1], raw_array[2], raw_array[3])

    def setup(self):

        # Set up vertices and indices
        self.terrainVertices = np.array(self.getVertices(self.vertexCount), dtype='float32')
        self.terrainIndices = np.array(self.getIndices(self.vertexCount), dtype='uint32')

        # Setup shaders
        self.shader = Shader(vertex_source="shaders/sensorData.vs", fragment_source="shaders/sensorData.fs")
        self.shader.use()

        # Set model matrix of terrain
        # self.model = Matrix44.from_translation(np.array(self.position))
        self.model = QMatrix4x4()
        self.model.scale(500.5, 1.0, 500.5)
        self.shader.setMat4("model", self.model)

        # Create Vertex Array Object
        self.__vao = glGenVertexArrays(1)
        glBindVertexArray(self.__vao)

        # Create Buffers and assign data
        bufs = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, bufs[0])
        glBufferData(GL_ARRAY_BUFFER, sizeof(ctypes.c_float) * len(self.terrainVertices), (ctypes.c_float * len(self.terrainVertices))(*self.terrainVertices), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(self.terrainIndices), (ctypes.c_uint * len(self.terrainIndices))(*self.terrainIndices), GL_STATIC_DRAW)
        
        # Turn on position attribute and set its size
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(ctypes.c_float), None)

        # Unbind buffers and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0);

        # Setup textures
        # self.colors = ReadTexture("textures/atacama_rgb.jpg")
        self.heightMap = ReadHeightMap("textures/atacama_height.png")
        self.shader.stop()



    


        