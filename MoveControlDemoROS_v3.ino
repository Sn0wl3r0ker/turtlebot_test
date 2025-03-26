#define Pin_Encoder_Right_A 2
#define Pin_Encoder_Right_B 3

#define Pin_Encoder_Left_A 20
#define Pin_Encoder_Left_B 21
//M1左輪   M2右輪
//     E2A----------------2
//     E2B----------------3
//     E1A----------------20
//     E1B----------------21

long theta_Right = 0, theta_Left = 0;
unsigned long currentMillis;
long previousMillis = 0;    // set up timers
float interval = 100;

//直流馬達----------TB6612腳位----------ArduinoU腳位
//                     PWMA-----------------4
//                     AIN1-----------------6
//                     AIN2-----------------5
//                     STBY-----------------7
//                     PWMB-----------------8
//                     BIN1-----------------10
//                     BIN2-----------------9
//                     
//                     GND------------------GND
//                     VM-------------------12V
//                     VCC------------------5V
//                     GND------------------GND
//直流馬達----------TB6612腳位----------ArduinoUNO腳位

//定義腳位
#define PWMA 4  
#define AIN1 6
#define AIN2 5
#define PWMB 8  
#define BIN1 10
#define BIN2 9
#define STBY 7  

int PwmA, PwmB;

#include <ros.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Int16MultiArray.h>

ros::NodeHandle nh;
// PWM命令回調函數
/*void pwmCallback(const std_msgs::Int16& msg) {
  int pwm_value1 = msg.data[0];
  int pwm_value2 = msg.data[1];
  SetPWM(1, -pwm_value1);;  // 設定PWM值來控制馬達轉速
  SetPWM(2, pwm_value2);
}*/
void pwmCallback(const std_msgs::Int16MultiArray& msg) {
  if (msg.data_length >= 2) {
    int pwm_1 = msg.data[0];
    int pwm_2 = msg.data[1];
    SetPWM(1, -pwm_1);   // 左馬達 // 設定PWM值來控制馬達轉速
    SetPWM(2, pwm_2);   // 右馬達
  }
}
// ROS訂閱者
//ros::Subscriber<std_msgs::Int16> pwm_sub("set_pwm", pwmCallback);
ros::Subscriber<std_msgs::Int16MultiArray> pwm_sub("set_pwm", pwmCallback);


void setup() {
  Serial2.begin(115200);
  nh.getHardware()->setBaud(115200);    // set baud rate to 115200
  nh.getHardware()->setPort(&Serial2);  //IMPORTANT!
  initEncoder();
  initMotor();
  previousMillis = millis();
  // 初始化ROS
  nh.initNode();
  nh.subscribe(pwm_sub);
}

/**************************************************************************
函數功能：設置指定馬達轉速
輸入参數：指定馬達motor，motor=1（2）代表馬達A（B）； 指定轉速pwm，大小範圍为0~255，代表停轉和全速
返回值：無
**************************************************************************/
void SetPWM(int motor, int pwm)
{
  if(motor==1&&pwm>=0)      //motor=1代表控制馬達A，pwm>=0則(AIN1, AIN2)=(1, 0)為正轉
  {
    digitalWrite(AIN1, 1);
    digitalWrite(AIN2, 0);
    analogWrite(PWMA, pwm);
  }
  else if(motor==1&&pwm<0)  //motor=1代表控制馬達A，pwm<0則(AIN1, AIN2)=(0, 1)為反轉
  {
    digitalWrite(AIN1, 0);
    digitalWrite(AIN2, 1);
    analogWrite(PWMA, -pwm);
  }
  else if(motor==2&&pwm>=0)   //motor=2代表控制馬達B，pwm>=0則(BIN1, BIN2)=(0, 1)為正轉
  {
    digitalWrite(BIN1, 0);
    digitalWrite(BIN2, 1);
    analogWrite(PWMB, pwm);
  }
  else if(motor==2&&pwm<0)    //motor=2代表控制馬達B，pwm<0則(BIN1, BIN2)=(1, 0)為反轉
  {
    digitalWrite(BIN1, 1);
    digitalWrite(BIN2, 0);
    analogWrite(PWMB, -pwm);
  }
}
unsigned char Read_Table[10] = {0};
void loop() 
{
// ROS處理
  nh.spinOnce();
  delay(10);  // 根據需要調整延遲
}

void initEncoder(){
  pinMode(Pin_Encoder_Right_A, INPUT_PULLUP);
  pinMode(Pin_Encoder_Right_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(Pin_Encoder_Right_A), doEncoder_Right_A, RISING);
  pinMode(Pin_Encoder_Left_A, INPUT_PULLUP);
  pinMode(Pin_Encoder_Left_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(Pin_Encoder_Left_A), doEncoder_Left_A, RISING);  
}
void forward(int s1,int s2){
    SetPWM(1, s1);
    SetPWM(2, -s2);
}
void right(int s1,int s2){
    SetPWM(1, s1);
    SetPWM(2, s2);
}
void left(int s1,int s2){
    SetPWM(1, -s1);
    SetPWM(2, -s2);
}
void back(int s1,int s2){
    SetPWM(1, -s1);
    SetPWM(2, s2);
}
void stopp(){
    SetPWM(1, 0);
    SetPWM(2, 0);
}


void initMotor(){
  //控制訊號初始化
  pinMode(AIN1, OUTPUT);//控制馬達A的方向，(AIN1, AIN2)=(1, 0)為正轉，(AIN1, AIN2)=(0, 1)為反轉
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);//控制馬達B的方向，(BIN1, BIN2)=(0, 1)為正轉，(BIN1, BIN2)=(1, 0)為反轉
  pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);//A馬達PWM
  pinMode(PWMB, OUTPUT);//B馬達PWM
  pinMode(STBY, OUTPUT);//TB6612致能,設置0則所有馬達停止,設置1才允許控制馬達

  //初始化TB6612馬達驅動模組
  digitalWrite(AIN1, 1);
  digitalWrite(AIN2, 0);
  digitalWrite(BIN1, 1);
  digitalWrite(BIN2, 0);
  digitalWrite(STBY, 1);
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

void doEncoder_Right_A() {
  if (digitalRead(Pin_Encoder_Right_B) == LOW)
    theta_Right = theta_Right - 1;
  else
    theta_Right = theta_Right + 1;
}//end of void doEncoder_Right_A()
//===============================================================================
inline void doEncoder_Left_A() {
  if (digitalRead(Pin_Encoder_Left_B) == LOW)
    theta_Left = theta_Left + 1;
  else
    theta_Left = theta_Left - 1;
}//end of void doEncoder_Left_A()
