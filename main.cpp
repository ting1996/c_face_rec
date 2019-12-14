#include <opencv2/opencv.hpp>


#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <experimental/filesystem>  
#include <map>
#include <string>
#include "UltraFace/UltraFace.hpp"
using namespace dlib;
using namespace std;



std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


int main()
{
    int capture_width = 640 ;
    int capture_height = 360 ;
    int display_width = 640 ;
    int display_height = 360 ;
    int framerate = 21 ;
    int flip_method = 2 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

    cv::Mat img, blod;

     frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    /* net_type net_detect;
    deserialize("mmod_human_face_detector.dat") >> net_detect;  */ 

    std::map<std::string, dlib::matrix<float,0,1>> data_faces;
    
    deserialize("data_faces.dat") >> data_faces;
    
    
    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.85);
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    std::cout << "Hit ESC to exit" << "\n" ;
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();

        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        faces.clear();

        win.clear_overlay();
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0));
            auto shape = sp(matrix,rect);
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        
        
        win.set_image(cimg);
        cout << "Detection delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
        m_StartTime = std::chrono::system_clock::now();
        
        
        /* matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        win.clear_overlay();
        win.set_image(matrix);
        faces.clear();
        //auto dets = net_detect(matrix);
        for (auto&& face : detector(matrix))
        {
            auto shape = sp(matrix, face);
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(face);
        } */

        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            continue;
        }
        
        std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
        
        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            for(auto& x:data_faces )
            {
                if (length(face_descriptors[i]-x.second ) < 0.4)
                    cout << x.first <<": "<< length(face_descriptors[i]-x.second )<< endl;
            }
        }
        cout <<" Rec delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
    }

    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}