#include <opencv2/opencv.hpp>

#include "mysql_connection.h"
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/sqlite.h>
#include <map>
#include <string>
#include "UltraFace/UltraFace.hpp"

#include <math.h>
using namespace dlib;
using namespace std;



std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
}

bool table_exists (
    database& db,
    const std::string& tablename
)
{
    // Sometimes you want to just run a query that returns one thing.  In this case, we
    // want to see how many tables are in our database with the given tablename.  The only
    // possible outcomes are 1 or 0 and we can do this by looking in the special
    // sqlite_master table that records such database metadata.  For these kinds of "one
    // result" queries we can use the query_int() method which executes a SQL statement
    // against a database and returns the result as an int.
    return query_int(db, "select count(*) from sqlite_master where name = '"+tablename+"'")==1;
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



void updata_mysql(sql::PreparedStatement *ps, const string name, matrix<rgb_pixel>& img)
{
    cv::Mat tmp = toMat(img);

    std::vector<unsigned char> buff(22500);
    cv::imencode(".png", tmp, buff,{cv::IMWRITE_PNG_STRATEGY_DEFAULT,1});
    unsigned char* buffimg = &buff[0];

    std::string value(reinterpret_cast<char*>(buffimg),22500);
    std::istringstream tmp_blob(value);
    ps->setBlob(2,&tmp_blob);
    ps->setString(1,name);
    ps->executeUpdate();     
}

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2){
    // treat two empty mat as identical as well
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    // if dimensionality of two mat is not identical, these two mat is not identical
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
        return false;
    }
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nz = cv::countNonZero(diff);
    return nz==0;
}


double get_blinking_ratio(int eye_points[],dlib::full_object_detection facial_landmarks)
{
    cv::Point left_point (facial_landmarks.part(eye_points[0]).x(), facial_landmarks.part(eye_points[0]).y());
    cv::Point right_point (facial_landmarks.part(eye_points[3]).x(), facial_landmarks.part(eye_points[3]).y());
    cv::Point top1 (facial_landmarks.part(eye_points[1]).x(),facial_landmarks.part(eye_points[1]).y());
    cv::Point top2 (facial_landmarks.part(eye_points[2]).x(),facial_landmarks.part(eye_points[2]).y());
    cv::Point bot1 (facial_landmarks.part(eye_points[5]).x(),facial_landmarks.part(eye_points[5]).y());
    cv::Point bot2 (facial_landmarks.part(eye_points[4]).x(),facial_landmarks.part(eye_points[4]).y());
    cv::Point center_top = (top1 + top2)/2;
    cv::Point center_bottom = (bot1 + bot2)/2;

    //hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)
    //ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2)

    double hor_line_lenght = hypot((left_point.x - right_point.x), (left_point.y - right_point.y));
    double ver_line_lenght = hypot((center_top.x - center_bottom.x), (center_top.y - center_bottom.y));

    
    return hor_line_lenght / ver_line_lenght;
} 
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

    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    int r_eye_points[] = {42, 43, 44, 45, 46, 47};
    int l_eye_poits[] = {36, 37, 38, 39, 40, 41};
    // And finally we load the DNN responsible for face recognition.
    
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    

    //Load know faces
    std::map<std::string, dlib::matrix<float,0,1>> data_faces;   
    deserialize("data_faces.dat") >> data_faces;
    
    //Template database

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    std::cout << "Hit ESC to exit" << "\n" ;

    cv::Mat img;
    
    while(true)
    {
        

    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
        //cap.read(img);
        //if(skip)
        //    continue;
      
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);

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
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
            auto shape = sp(matrix,rect);
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);
           
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        
        
        win.set_image(matrix);
        cout << "Detection delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
        m_StartTime = std::chrono::system_clock::now();
        
        
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
                //double tmp_distance = length(face_descriptors[i]-x.second );
                if (length(face_descriptors[i]-x.second ) < 0.5)
                {
                    
                    cout<<x.first<<": "<<length(face_descriptors[i]-x.second )<<endl;
                }
            }
        }
        cout <<" Rec delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
        
    }
    cap.release();
    return 0;
}