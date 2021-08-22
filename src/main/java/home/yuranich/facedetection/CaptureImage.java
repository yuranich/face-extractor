package home.yuranich.facedetection;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

import javax.swing.*;
import java.io.IOException;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class CaptureImage {
    private CvHaarClassifierCascade classifier = null;
    private CvMemStorage storage = null;
    private Exception exception = null;
    private IplImage grabbedImage = null, grayImage = null, smallImage = null;
    private CanvasFrame canvas = new CanvasFrame("Webcam");

    // use default camera
    private OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
    private OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

    private CvSeq faces = null;

    private static final String CASCADE_FILE = "./haarcascade_frontalface_default.xml";


    public static void main(String[] args) {
        CaptureImage ci = new CaptureImage();
        ci.init();
        ci.loop();
    }

    private void init() {
        try {
            classifier = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
            if (classifier.isNull()) {
                throw new IOException("Could not load the classifier file.");
            }

            storage = CvMemStorage.create();
        } catch (Exception e) {
            // print stack trase
            e.printStackTrace();
        }
    }

    private void loop() {
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        try {
            grabber.start();

            // get framerate
            double frameRate = grabber.getFrameRate();
            long wait = (long) (1000 / (frameRate == 0 ? 10 : frameRate));

            // keep capturing
            while (true) {
                Thread.sleep(wait);
                grabbedImage = converter.convert(grabber.grab());

                // grayscale
                grayImage  = IplImage.create(grabbedImage.width(),   grabbedImage.height(),   IPL_DEPTH_8U, 1);
                // resize image
                smallImage = IplImage.create(grabbedImage.width()/2, grabbedImage.height()/2, IPL_DEPTH_8U, 1);

                cvClearMemStorage(storage);
                cvCvtColor(grabbedImage, grayImage, CV_BGR2GRAY);
                cvResize(grayImage, smallImage, CV_INTER_AREA);
                faces = cvHaarDetectObjects(smallImage, classifier, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);

                if (faces != null) {
                    System.out.println(faces.total());
                }

                int faces_num = faces.total();

                for(int i = 0; i < faces_num; i++){
                    CvRect r = new CvRect(cvGetSeqElem(faces, i));
                    // show retangle around detected faces.
                    cvRectangle (
                            smallImage,
                            cvPoint(r.x(), r.y()),
                            cvPoint(r.width() + r.x(), r.height() + r.y()),
                            CvScalar.RED,
                            2,
                            CV_AA,
                            0);

                }

                // show grabbed image
                if (grabbedImage != null) {
                    canvas.showImage(converter.convert(smallImage));
                }
            }

            // show stack trace
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}