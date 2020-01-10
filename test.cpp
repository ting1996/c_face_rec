#include <ctime>
#include <iostream>

#include <thread>
#include <atomic>

#include <unistd.h>

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
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <map>
#include <string>
#include <opencv2/imgcodecs.hpp>
using namespace std;
using namespace sql;
using namespace dlib;
atomic<int> status(-3);

void check_status(const char* host,const char* user,const char* password,const char* schema, int id_device)
{
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect(host, user, password); //IP Address, user name, password
        con->setSchema(schema);
        stmt = con->createStatement();
        while(true)
        {
            res = stmt->executeQuery("SELECT * FROM device WHERE id_device = " + std::to_string(id_device) + ";");
            res->next();
                
            //cout << res->getString("status") << endl;
            status.store(res->getInt("status"), std::memory_order_release);
            if(res->getInt("status") == -2)
            {
                break;
            }

            usleep(1000000);
        }
        delete res;
        delete stmt;
        delete con;
    } 
    catch (sql::SQLException &e) {
        status.store(4, std::memory_order_release);
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
}

void upload_image(const char* host,const char* user,const char* password,const char* schema, const char* imgpath)
{
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;
        sql::PreparedStatement *ps;
        cv::Mat tmp = cv::imread(imgpath,cv::IMREAD_UNCHANGED);

        std::vector<unsigned char> buff(256*256);
        cv::imencode(".png", tmp, buff,{cv::IMWRITE_PNG_COMPRESSION,9});
        unsigned char* buffimg = &buff[0];
        /* Create a connection */
        driver = get_driver_instance();
        
        con = driver->connect(host, user, password); //IP Address, user name, password
        con->setSchema(schema);

        ps = con->prepareStatement("INSERT INTO diem_danh(ID_employ, img) VALUES (?,?)");
        //stmt = con->createStatement();
        //std::string tmp_str = std::string("INSERT INTO dem_danh(ID_employ, img) VALUES (ID_123,") + std::string(reinterpret_cast<char*>(buffimg)) + ")";
        std::string value(reinterpret_cast<char*>(buffimg),256*256);
        std::istringstream tmp_blob(value);
        ps->setBlob(2,&tmp_blob);
        ps->setString(1,"ID_123");
        //stmt->execute(tmp_str);
        ps->executeUpdate();
        con->commit();

        delete res;
        delete stmt;
        delete con;
    } 
    catch (sql::SQLException &e) {
        status.store(4, std::memory_order_release);
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
}

void test_1(string& a)
{
    int a_int = 0;
    while(true)
    {
        a_int++;
        a = std::to_string(a_int);
    }
}
int main() {
    /* thread t1(upload_image, "tcp://35.247.163.122:3306",
                            "ting199x",
                            "97654321",
                            "abc_db",
                            "logo.png");
    t1.join(); */
    /* thread t1(check_status, "tcp://35.247.163.122:3306",
                            "ting199x",
                            "97654321",
                            "abc_db",
                            1);
    while(true)
    {
        int tmp_status = status.load(std::memory_order_acquire);
        cout<<tmp_status<<endl;
        if(tmp_status == -2)
            break;
    }
    t1.join();
 */
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);

    while(true)
    {
        t = std::time(0);
        cout<<now->tm_sec<<endl;
        usleep(1000000);

    }
}