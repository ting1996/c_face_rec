#include <iostream>
#include "JetsonGPIO.h"
#include <map>
#include <unistd.h>
using namespace std;
void blink_led(int a, int &b)
{

    GPIO::output(a, b);
	usleep(1000*1000/);
	b = (b - 1)* -1;
	GPIO::output(a, b);
}
int main(){
	cout << "model: "<< GPIO::model << endl;
	cout << "lib version: " << GPIO::VERSION << endl;
	cout << GPIO::JETSON_INFO << endl;

	map<int,int> channels;
	channels[16] = 0;
	GPIO::setmode(GPIO::BOARD);
	GPIO::setup(16,GPIO::OUT, channels[16]);
	
	while(true)
	{
		blink_led(16,channels[16] );
	}
	
	
	
	

	GPIO::cleanup();	

	cout << "end" << endl;	
	return 0;
}