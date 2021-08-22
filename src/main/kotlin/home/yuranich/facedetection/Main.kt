package home.yuranich.facedetection

import home.yuranich.facedetection.FaceUtils
import java.io.File

fun main(args: Array<String>) {
    if (args.size != 2) {
        throw IllegalArgumentException("Expected parameters: <input folder> <output folder>")
    }
    FaceDetection.loadModel()
    val file = File(args[0])

    var counter = 0
    file.walkBottomUp()
        .onEnter { true }
        .filter { it.isFile}
        .map {
            FaceUtils.readImage(it)
        }
        .flatMap {
            val faces = FaceUtils.detectFaces(it)
            faces.zip( IntArray(faces.size) { ++counter }.asIterable()).asSequence()
        }.forEach {
            FaceUtils.writeFace(it.first, it.second, args[1])
        }

}

