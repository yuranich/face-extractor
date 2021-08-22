package home.yuranich.facedetection

import home.yuranich.facedetection.FaceDetection
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgcodecs.imwrite
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File

object FaceUtils {

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

    fun writeFace(face: Mat, num: Int, absoluteFolderPath: String) {
        println("Writing face: $num")
        imwrite("${absoluteFolderPath}${File.separator}face$num.jpg", face)
    }


}