// IF you are unable to upload the file to Arduino then uncomment the "Adafruit_SSD1306.h"
// and "SimpleDHT.h" 
//#include <Adafruit_SSD1306.h>
//#include <SimpleDHT.h>
#include <LiquidCrystal.h>

#define led 6
#define led1 7
#define led2 8
#define led3 9
#define led4 10
int data, flag = 2;

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup()
{
  pinMode(led, OUTPUT);
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  pinMode(led4, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  digitalWrite(led, LOW);
  digitalWrite(led1, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  lcd.begin(16, 2);                       
  lcd.setCursor (0,0);                   
  lcd.print("                ");
  lcd.setCursor (0,1);
  lcd.print("                ");
}

void loop()
{
  while( Serial.available() )
  {
    data = Serial.read();

    if (data == '1')
    {
      flag = 1;
    }
    else if(data == '0')
    {
      flag = 0;
    }
  }
  if(flag == 1)
    {
      lcd.setCursor (0,0);
      lcd.print(" Welcome Boss!!  ");
      digitalWrite(led, HIGH);
      digitalWrite(led1, LOW);
      digitalWrite(led2, LOW);
      digitalWrite(led3, LOW);
      digitalWrite(led4, LOW);
      delay(200);
      digitalWrite(led, LOW);
      digitalWrite(led1, HIGH);
      digitalWrite(led2, LOW);
      digitalWrite(led3, LOW);
      digitalWrite(led4, LOW);
      delay(200);
      digitalWrite(led, LOW);
      digitalWrite(led1, LOW);
      digitalWrite(led2, HIGH);
      digitalWrite(led3, LOW);
      digitalWrite(led4, LOW);
      delay(200);
      digitalWrite(led, LOW);
      digitalWrite(led1, LOW);
      digitalWrite(led2, LOW);
      digitalWrite(led3, HIGH);
      digitalWrite(led4, LOW);
      delay(200);
      digitalWrite(led, LOW);
      digitalWrite(led1, LOW);
      digitalWrite(led2, LOW);
      digitalWrite(led3, LOW);
      digitalWrite(led4, HIGH);
      delay(200);
      digitalWrite(LED_BUILTIN, HIGH);
    }
     else if (flag == 0)
    {
      lcd.setCursor (0,0);
      lcd.print("Face Unidentified  ");
      digitalWrite(led, LOW);
      digitalWrite(led1, LOW);
      digitalWrite(led2, LOW);
      digitalWrite(led3, LOW);
      digitalWrite(led4, LOW);
      digitalWrite(LED_BUILTIN, LOW);
    }
}
