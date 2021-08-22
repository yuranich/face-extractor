package home.yuranich.facedetection

import home.yuranich.facedetection.FaceDetection
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgcodecs.imwrite
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File

object FaceUtils {

    fun extractorFace(file: File) {
        println("Reading path: ${file.absolutePath}")
        val image = imread(file.absolutePath)
        println("Loaded image: ${image}")
        val rects = FaceDetection.detectFaces(image)
        println("Detected ${rects.size()} faces.")
        val faces = (0 until rects.size()).map {
            val rect = rects[it]

            image.apply(rect)
        }
        if (faces.size == 1) {
            println("Extracted image: ${faces[0]}")
            imwrite("/home/yuranich/Pictures/Faces/face.jpg", faces[0])
        } else {
            println("No or too many faces found.")
        }
    }

    fun readImage(file: File): Mat {
        println("Reading path: ${file.absolutePath}")
        return imread(file.absolutePath)
    }

    fun detectFaces(image: Mat): List<Mat> {
        val rects = FaceDetection.detectFaces(image)
        println("Detected ${rects.size()} faces.")
        val faces = (0 until rects.size()).map {
            val rect = rects[it]

            image.apply(rect)
        }
        return faces
    }

    fun writeFace(face: Mat, num: Int) {
        println("Writing face: $num")
        imwrite("/home/yuranich/Pictures/Faces/face$num.jpg", face)
    }


}