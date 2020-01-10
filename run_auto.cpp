#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "JetsonGPIO.h"

#include <fstream>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <map>
#include <string>
#include "UltraFace/UltraFace.hpp"
#include <experimental/filesystem> 

#include <ctime>
#include <iostream>

#include <thread>
#include <atomic>

#include "mysql_connection.h"
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
using namespace dlib;
using namespace std;
namespace fs = std::experimental::filesystem;

atomic<int> status(-3);
atomic<int> command(0);

bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
{
	if(!ofs.is_open()){
		return false;
	}
	//ofs.write("150", sizeof(int));
   
	//ofs.write("150", sizeof(int));
    
	//ofs.write("16", sizeof(int));
   
	ofs.write((const char*)(out_mat.data), 67500);

	return true;
}


//! Save cv::Mat as binary
/*!
\param[in] filename filaname to save
\param[in] output cvmat to save
*/
bool SaveMatBinary(const std::string& filename, const cv::Mat& output){
	std::ofstream ofs(filename, std::ios::binary);
	return writeMatBinary(ofs, output);
}


//! Read cv::Mat from binary
/*!
\param[in] ifs input file stream
\param[out] in_mat mat to load
*/
bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
	if(!ifs.is_open()){
		return false;
	}
	
	/* int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	if(rows==0){
		return true;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int)); */

	in_mat.release();
	in_mat.create(150, 150, 16);
	ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

	return true;
}


//! Load cv::Mat as binary
/*!
\param[in] filename filaname to load
\param[out] output loaded cv::Mat
*/
bool LoadMatBinary(const std::string& filename, cv::Mat& output){
	std::ifstream ifs(filename, std::ios::binary);
	return readMatBinary(ifs, output);
}

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
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





bool check_internet()
{

    if (system("ping -c1 -s1 www.google.com"))
    {
        int tmp = status.load(std::memory_order_acquire);
        if(tmp != 4 || tmp != 5 || tmp != 6 || tmp != -2)
        {
            cout<<"There is no internet connection  \n";
            status.store(4, std::memory_order_release);
            return false;
        }
    }
    else
    {
        int tmp = status.load(std::memory_order_acquire);
        if(tmp != 0 || tmp != -1 || tmp != 1 || tmp != 2 || tmp != 3|| tmp != -2)
        {
            cout<<"There is internet connection  \n";
            status.store(0, std::memory_order_release);
            return true;
        }
    }
    usleep(10000000);
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

std::vector<int> cvtStrTime2Int(const std::string& str, const std::string& delim)
{
    std::vector<string> vt_tmp = split(str,delim);
    return std::vector<int>{std::stoi(vt_tmp[0]), std::stoi(vt_tmp[1]), std::stoi(vt_tmp[2])};
    
}

bool isSmaller(std::vector<int> & a, std::vector<int> & b)
{
    std::vector<int> tmp = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    if(tmp[0] < 0 )
        return false;
    if(tmp[0] > 0 )
        return true;

    if(tmp[1] < 0)
        return false;
    if(tmp[1] > 0)
        return true;

    if(tmp[2] < 0)
        return false;
    if(tmp[2] > 0)
        return true;

    return false;
}
bool check_time_work(const string& start_at, const string& end_at, const string & now)
{
    std::vector<int> vt_start_at = cvtStrTime2Int(start_at,":");
    std::vector<int> vt_end_at   = cvtStrTime2Int(end_at,":");
    std::vector<int> vt_now      = cvtStrTime2Int(now,":");

    if(vt_end_at[0] < vt_start_at[0])
    {
        std::vector<int> midnight_24{24,0,0};
        std::vector<int> midnight_0{0,0,0};
         if(isSmaller(vt_start_at, vt_now) && isSmaller(vt_now,midnight_24))
            return true;
         if(isSmaller( midnight_0, vt_now) && isSmaller(vt_now,vt_end_at))
            return true;
    }
    else
    {
        if(isSmaller(vt_start_at, vt_now) && isSmaller(vt_now,vt_end_at))
            return true;
    }
    return false;

}

void get_data_from_mysql(   string& host,
                            string& user,
                            string& password,
                            string& schema, 
                            int& id_device,
                            string& rec_start_at,
                            string& rec_end_at,
                            string& updata_start_at,
                            string& updata_end_at)
{
    
    while(true)
    {
        try {
            
            if(command.load(std::memory_order_acquire) == 0)
            {
                /* Create a connection */
                sql::Driver *driver;
                sql::Connection *con;
                sql::Statement *stmt;
                sql::ResultSet *res;
                driver = get_driver_instance();
                con = driver->connect(host, user, password); //IP Address, user name, password
                con->setSchema(schema);
                stmt = con->createStatement();
                
                
                res = stmt->executeQuery("SELECT * FROM device WHERE id_device = " + std::to_string(id_device) + ";");
                res->next();
                //get command
                cout << res->getString("command") << endl;
                command.store(res->getInt("command"), std::memory_order_release);
                int int_command = res->getInt("command");
                
                if(int_command == 1)
                    break;
        
                //get time work
                if(int_command != 0)
                    continue;
                
                res = stmt->executeQuery("SELECT * FROM time_work WHERE id_device = " + std::to_string(id_device) + ";");
                while(res->next())
                {
                    if(res->getInt("status") == 1) //face rec
                    {
                        rec_start_at = res->getString("start_at");
                        rec_end_at  = res->getString("end_at");
                    }
                    if(res->getInt("status") == 3) //updata to sql
                    {
                        updata_start_at = res->getString("start_at");
                        updata_end_at = res->getString("end_at");
                    }
                }
                
                //check time work
                std::time_t t = std::time(0);
                std::tm* now = std::localtime(&t);
                char* char_tmp;
                
                
                string str_now = to_string(now->tm_hour) + ":" + to_string(now->tm_min) + ":" + to_string(now->tm_sec);
                cout<<str_now<<": time"<<endl;

                if(check_time_work(rec_start_at,rec_end_at,str_now))
                    status.store(1, std::memory_order_release);
                else if(check_time_work(updata_start_at,updata_end_at,str_now))
                    status.store(3, std::memory_order_release);
                else 
                {
                    status.store(-4, std::memory_order_release);
                }
                delete res;
                delete stmt;
                delete con;
            }
            
            std::this_thread::sleep_for (std::chrono::seconds(1));
        } 
        catch (sql::SQLException &e) {
            status.store(1, std::memory_order_release);
            command.store(0, std::memory_order_release);
            cout << "# ERR: " << e.what();
            cout << " (MySQL error code: " << e.getErrorCode();
            cout << ", SQLState: " << e.getSQLState() << " )" << endl;
        }

    }
    
}

void update_status_command( string& host,
                            string& user,
                            string& password,
                            string& schema, 
                            int& id_device
                            )
{
    
    while(true)
    {
        try {
            cout<<"loop update"<<endl;
            /* Create a connection */
            sql::Driver *driver;
            sql::Connection *con;
            sql::PreparedStatement *ps;

            driver = get_driver_instance();
            con = driver->connect(host, user, password); //IP Address, user name, password
            con->setSchema(schema);
            
            ps = con->prepareStatement("UPDATE device SET  status = ?, check_alive = ? WHERE (id_device = " + std::to_string(id_device) + ");");
            
           
            ps->setInt(1,status.load(std::memory_order_acquire));
            ps->setInt(2, 1);
            ps->executeUpdate();
            if(command.load(std::memory_order_acquire) == -1)
            {
                ps = con->prepareStatement("UPDATE device SET  command = ? WHERE (id_device = " + std::to_string(id_device) + ");");
                ps->setInt(1,0);
                ps->executeUpdate();
                break;
            }
            if(command.load(std::memory_order_acquire) == -2)
            {
                command.store(0, std::memory_order_release);
                ps = con->prepareStatement("UPDATE device SET  command = ? WHERE (id_device = " + std::to_string(id_device) + ");");
                ps->setInt(1,0);
                ps->executeUpdate();            
            }
            
            delete con;
            delete ps;

            std::this_thread::sleep_for (std::chrono::seconds(2));
        } 
        catch (sql::SQLException &e) {
            cout << "# ERR: " << e.what();
            cout << " (MySQL error code: " << e.getErrorCode();
            cout << ", SQLState: " << e.getSQLState() << " )" << endl;
        }

    }
    
}

bool updata_mysql_server(   string& host,
                            string& user,
                            string& password,
                            string& schema,
                            const string name, 
                            cv::Mat& img, 
                            double distace, 
                            const string date)
{
    

    try {
            
            // Create a connection 
            sql::Driver *driver;
            sql::Connection *con;
            sql::PreparedStatement *ps;

            driver = get_driver_instance();
            con = driver->connect(host, user, password); //IP Address, user name, password
            con->setSchema(schema);
            
            ps = con->prepareStatement("INSERT INTO diem_danh(id_employ,img, distance, date) VALUES (?,?,?,?)");
            ps->setString(1,name);
            ps->setDouble(3,distace);
            ps->setString(4,date);


            std::vector<unsigned char> buff(67500);//22500
            cv::imencode(".png", img, buff,{cv::IMWRITE_PNG_COMPRESSION,9});
            unsigned char* buffimg = &buff[0];

            std::string value(reinterpret_cast<char*>(buffimg),67500);
            std::istringstream tmp_blob(value);
            ps->setBlob(2,&tmp_blob);
            ps->executeUpdate();
            delete ps;
            
            delete con;  
        } 
        catch (sql::SQLException &e) {
            cout << "# ERR: " << e.what();
            cout << " (MySQL error code: " << e.getErrorCode();
            cout << ", SQLState: " << e.getSQLState() << " )" << endl;
            return false;
        }
    return true;
}

void updata_mysql_local(sql::Statement *stmt, sql::PreparedStatement *ps, const string name, matrix<rgb_pixel>& img, double& distance)
{
    sql::ResultSet *res;
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    ps->setString(3,to_string(now->tm_year+ 1900) + "-" + to_string(now->tm_mon + 1) + "-" + to_string(now->tm_mday) + " " + to_string(now->tm_hour) + ":" + to_string(now->tm_min) + ":" + to_string(now->tm_sec));//to_string(now->tm_year) + "-" + to_string(now->tm_mon + 1) + "-" + to_string(now->tm_mday) + " " + to_string(now->tm_hour) + ":" + to_string(now->tm_min) + ":" + to_string(now->tm_sec)
    res = stmt->executeQuery("SELECT AUTO_INCREMENT FROM  INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'abc_db' AND   TABLE_NAME   = 'diem_danh';");
    ps->setDouble(2,distance);
    ps->setString(1,name);
    
    ps->executeUpdate();

    res->next();
    SaveMatBinary("img_data/" + res->getString("AUTO_INCREMENT"), toMat(img));
    delete res;

}
void blink_led(int& a)
{
    
}

void addClk(int& a,int& b, int& c )
{
    b += c;
    if (b >= 60 )
    {
        b -= 60;
        a++;
    } 
}

void subClk(int& a,int& b, int& c )
{
    b += c;
    if (b < 0 )
    {
        b += 60;
        a--;
    } 
}

string int2StrClk(int a, int b,int c)
{
    if(c < 0)
        subClk(a,b,c);
    else
        addClk(a,b,c);
    return to_string(a) + ":" + to_string(b) + ":00"; 
}

int main()
{
    //Thread check internet
    thread check_internet_thread(check_internet);

    //Init config
    std::string initsql;
    std::string host;
    std::string user;
    std::string password;
    std::string schema;

    std::string rec_start_at;
    std::string rec_end_at;

    std::string updata_start_at;
    std::string updata_end_at;


     if (!fs::exists("initsql.txt"))
     {
        host = "tcp://35.247.163.122:3306";
        user = "ting199x";
        password = "97654321";
        schema = "abc_db";

        rec_start_at = "nah";
        rec_end_at = "nah";
        updata_start_at = "nah";
        updata_end_at = "nah";

        initsql = host + "," + user + "," + password + "," + schema + "," + rec_start_at + "," + rec_end_at + "," + updata_start_at +  "," + updata_end_at;
        serialize("initsql.txt") << initsql;
     }
     else
     {
        deserialize("initsql.txt") >> initsql;
        std::vector<std::string> tmp_initsql = split(initsql,",");

        host = tmp_initsql[0];
        user = tmp_initsql[1];
        password = tmp_initsql[2];
        schema = tmp_initsql[3];

        rec_start_at = tmp_initsql[4];
        rec_end_at = tmp_initsql[5];
        updata_start_at = tmp_initsql[6];
        updata_end_at = tmp_initsql[7];
     }
    int id_device = 1;

    //init face rec and data
    int capture_width = 640 ;
    int capture_height = 360 ;
    int display_width = 640 ;
    int display_height = 360 ;
    int framerate = 24 ;
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
    }


    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::map<std::string, dlib::matrix<float,0,1>> data_faces;
    if (!fs::exists("data_faces.dat"))
    {
        serialize("data_faces.dat") << data_faces;
    }
    else
    {
        deserialize("data_faces.dat") >> data_faces;
    }

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.85);
    cv::Mat img, blod;
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    int cout_img = 0;
    std::vector<matrix<rgb_pixel>> array_face;
    //Init led
    std::map<int,int> channels;
	channels[16] = 0;
    channels[18] = 0;
	GPIO::setmode(GPIO::BOARD);
	GPIO::setup(16, GPIO::OUT, channels[16]);
    GPIO::setup(18, GPIO::OUT, channels[18]);

    //Wait to check internet
    while(status.load(std::memory_order_acquire) == -3);
    check_internet_thread.join();


    thread get_data_from_mysql_thread (get_data_from_mysql, std::ref(host),
                                                            std::ref(user),
                                                            std::ref(password),
                                                            std::ref(schema),
                                                            std::ref(id_device),
                                                            std::ref(rec_start_at),
                                                            std::ref(rec_end_at),
                                                            std::ref(updata_start_at),
                                                            std::ref(updata_end_at)
                                                            );
    thread update_status_command_thread(update_status_command, std::ref(host),
                                                            std::ref(user),
                                                            std::ref(password),
                                                            std::ref(schema),
                                                            std::ref(id_device));

    int int_command;
    int int_status;
    //init local sql
    sql::Driver *driver_local;
    sql::Connection *con_local;
    sql::Statement *stmt_local;
    sql::PreparedStatement *ps_local;

    driver_local = get_driver_instance();
                    
    con_local = driver_local->connect("localhost", "ting199x", "97654321"); //IP Address, user name, password
    con_local->setSchema("abc_db");
    stmt_local = con_local->createStatement();
    ps_local = con_local->prepareStatement("INSERT INTO diem_danh(id_employ, distance, date) VALUES (?,?,?)");

    while(int_command != 1)
    {
        int_command = command.load(std::memory_order_acquire);
        int_status = status.load(std::memory_order_acquire);
        while(int_command == 0 && int_status == 1)
        {
            int_command = command.load(std::memory_order_acquire);
            int_status = status.load(std::memory_order_acquire);
            // Do face_rec
            
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
                image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
                auto shape = sp(matrix,rect);
                dlib::matrix<rgb_pixel> face_chip;
                extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
                faces.push_back(move(face_chip));
                win.add_overlay(orect);
            }
            
            
            win.set_image(cimg);
            cout << "Detection delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
            m_StartTime = std::chrono::system_clock::now();
            
            
            if (faces.size() == 0)
            {
                cout << "No faces found in image!" << endl;
                continue;
            }
            GPIO::output(16, 1);
            std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
            

            thread updata_mysql_local_thread[faces.size()];
            for (size_t i = 0; i < face_descriptors.size(); ++i)
            {
                string name = "";
                double distance = 1;
                for(auto& x:data_faces )
                {
                    double tmp_distance = length(face_descriptors[i]-x.second);
                    if (tmp_distance < 0.55)
                    {
                        if(distance > tmp_distance)
                        {
                            name = x.first;
                            distance = tmp_distance;
                        }
                    }
                }
                if(distance != 1)
                {
                    cout << name <<": "<< distance<< endl;
                    updata_mysql_local_thread[i] = thread(updata_mysql_local, std::ref(stmt_local), std::ref(ps_local), name,std::ref(faces[i]),std::ref(distance));
                }
            }
            for(int i = 0;i <faces.size(); i++ )
            {
                if(updata_mysql_local_thread[i].joinable())
                    updata_mysql_local_thread[i].join();
            }
            GPIO::output(16, 0);
            cout <<" Rec delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
        

        }
        while(int_command == 0 && int_status == 3)
        {
            int_command = command.load(std::memory_order_acquire);
            int_status = status.load(std::memory_order_acquire);
            //Get name and day
            cout<<"1 up"<<endl;
            sql::ResultSet *res_local;
            res_local = stmt_local->executeQuery("SELECT * FROM diem_danh limit 1;");
            if(res_local->next())
            {
                
                string tmp_name = res_local->getString("id_employ");
                string tmp_date = res_local->getString("date");
                tmp_date = split(tmp_date," ")[0];

                int index_start_day = 0;
                int index_end_day = 0;

                int start_hours = 7;
                int start_min = 0;
                int end_hours = 18;
                int end_min = 0;
                int step_min_per_loop = 10;

                while(start_hours != end_hours || start_min != end_min)
                {
                    string start_str = tmp_date + " "+ to_string(start_hours) + ":" + to_string(start_min) + ":" + "00";
                    string end_str = tmp_date + " "+ int2StrClk(start_hours,start_min,step_min_per_loop);
                    addClk(start_hours,start_min,step_min_per_loop);
                    cout<<start_str;
                    cout<<end_str<<endl;
                    res_local = stmt_local->executeQuery("select * from diem_danh where distance = (SELECT MIN(distance) FROM diem_danh where id_employ = '" + tmp_name + "' and date between '" + start_str + "' and '" + end_str + "');");
                    if(res_local->next())
                    {
                        index_start_day = res_local->getInt("id");
                        if(index_start_day != 0)
                            break;
                    }
                    
                }

                step_min_per_loop = -10;
                while(start_hours != end_hours || start_min != end_min)
                {
                    string end_str = tmp_date + " "+ to_string(end_hours) + ":" + to_string(end_min) + ":" + "00";
                    string start_str = tmp_date + " "+ int2StrClk(end_hours,end_min,step_min_per_loop);
                    subClk(end_hours,end_min,step_min_per_loop);
                    cout<<start_str;
                    cout<<end_str<<endl;
                    res_local = stmt_local->executeQuery("select * from diem_danh where distance = (SELECT MIN(distance) FROM diem_danh where id_employ = '" + tmp_name + "' and date between '" + start_str + "' and '" + end_str + "');");
                    if(res_local->next())
                    {
                        index_end_day = res_local->getInt("id");
                        if(index_end_day != 0)
                            break;
                    }
                }
                cout << index_start_day <<" "<<index_end_day<<endl;

                if(index_start_day != 0)
                {
                    res_local = stmt_local->executeQuery("SELECT * FROM diem_danh where id = " + to_string(index_start_day) + ";");
                    res_local->next();
                    cv::Mat xhip;
                    LoadMatBinary("img_data/" + to_string(index_start_day),xhip);
                    updata_mysql_server(host,user,password,schema,tmp_name,xhip,res_local->getDouble("distance"), res_local->getString("date"));
                }

                if(index_end_day != 0)
                {
                    res_local = stmt_local->executeQuery("SELECT * FROM diem_danh where id = " + to_string(index_end_day) + ";");
                    res_local->next();
                    cv::Mat xhip;   
                    LoadMatBinary("img_data/" + to_string(index_start_day),xhip);
                    updata_mysql_server(host,user,password,schema,tmp_name,xhip,res_local->getDouble("distance"), res_local->getString("date"));
                }

                //Delete all data in mysql local
                /* ps_local = con->prepareStatement("delete from diem_danh where id = ? and date between ? and ?;");
                ps_local->setString(1,tmp_name);
                ps_local->setString(2,tmp_date + " 0:0:0");
                ps_local->setString(3,tmp_date + " 23:59:59");
                ps_local->executeUpdate(); */

            }
            else
            {
                //delete all file in img_data 
            }
            delete res_local;
            // Do face_updata
        }
        if(int_command == 2)
        {
            //Do some thing face_data
            status.store(2, std::memory_order_release);
            string name =  " ";
            sql::Driver *driver;
            sql::Connection *con;
            sql::Statement *stmt;
            sql::ResultSet *res;
            try {
                
                /* Create a connection */
                driver = get_driver_instance();
                con = driver->connect(host, user, password); //IP Address, user name, password
                con->setSchema(schema);
                stmt = con->createStatement();
                res = stmt->executeQuery("SELECT * FROM device WHERE id_device = " + std::to_string(id_device) + ";");
                while(res->next())
                {
                    name = res->getString("tmp_id_employ");
                }
           
            } 
            catch (sql::SQLException &e) {
                cout << "# ERR: " << e.what();
                cout << " (MySQL error code: " << e.getErrorCode();
                cout << ", SQLState: " << e.getSQLState() << " )" << endl;
            }
            delete res;
            delete stmt;
            delete con;

            if(name == " " || name == "")
                continue;
            array_face.clear();
            while(true)
            {
                if (!cap.read(img)) {
                std::cout<<"Capture read error"<<std::endl;
                break;
                }
                
                cv_image<bgr_pixel> cimg(img);

                matrix<rgb_pixel> matrix;
                assign_image(matrix, cimg);
                win.clear_overlay();
                win.set_image(matrix);
                faces.clear();
                
                for (auto face : detector(matrix))
                {
                    auto shape = sp(matrix, face);
                    dlib::matrix<rgb_pixel> face_chip;
                    extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
                    faces.push_back(move(face_chip));
                    // Also put some boxes on the faces so we can see that the detector is finding
                    // them.
                    win.add_overlay(face);
                }

                if (faces.size() == 0)
                {
                    cout << "No faces found in image!" << endl;
                    continue;
                }

                cout_img++;
                cout << cout_img <<endl;
                if (cout_img == 100)
                    break;
                array_face.push_back(faces[0]);
                
            }
            data_faces.erase(name);
            data_faces.insert(std::pair<std::string, dlib::matrix<float,0,1>>(name,mean(mat(net(array_face)))));
            cout << data_faces[name];
            serialize("data_faces.dat") << data_faces;

            //When done
            command.store(-2, std::memory_order_release);
        }
        
    }
    delete con_local;
    delete stmt_local;
    delete ps_local;
    command.store(-1, std::memory_order_release);
    cout<< "out main"<<endl;
    get_data_from_mysql_thread.join();
    update_status_command_thread.join();
}