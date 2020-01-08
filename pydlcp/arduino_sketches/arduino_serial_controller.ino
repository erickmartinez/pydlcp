#include <CmdBuffer.hpp>
#include <CmdCallback.hpp>
#include <CmdParser.hpp>
#include <SPI.h>
#include "Adafruit_MAX31855.h"

#define P1 2
#define P2 3
#define P3 4
#define P4 5
#define P5 6
#define P6 7
#define P7 8
#define P8 9
#define PIN_KEITHLEY A0
#define PIN_FAN A1
#define MAXCS   10

CmdCallback<3> cmdCallback;
char strHello[]     = "HELLO";
char strTogglePin[] = "TOGGLE";
char strReadTemp[]  = "TEMP";

// initialize the Thermocouple
Adafruit_MAX31855 thermocouple(MAXCS);

void setup()
{

  pinMode(P1, OUTPUT);
  pinMode(P2, OUTPUT);
  pinMode(P3, OUTPUT);
  pinMode(P4, OUTPUT);
  pinMode(P5, OUTPUT);
  pinMode(P6, OUTPUT);
  pinMode(P7, OUTPUT);
  pinMode(P8, OUTPUT);
  pinMode(PIN_KEITHLEY, OUTPUT);
  pinMode(PIN_FAN, OUTPUT);

  digitalWrite(PIN_KEITHLEY, HIGH);
  
  Serial.begin(115200);

  cmdCallback.addCmd(strHello,    &functHello);
  cmdCallback.addCmd(strTogglePin,&functTogglePin);
  cmdCallback.addCmd(strReadTemp, &functTemp);
}

void loop()
{
  CmdBuffer<32> myBuffer;
  CmdParser     myParser;

  cmdCallback.loopCmdProcessing(&myParser, &myBuffer, &Serial);
}

void functHello(CmdParser *myParser) { Serial.println("Howdy!"); }

void functTemp(CmdParser *myParser) 
{
  double c = thermocouple.readCelsius();
  if (!isnan(c)) {
    Serial.println(c);
  } else {
    Serial.println("ERROR");
  }
}

void functTogglePin(CmdParser *myParser)
{
  // Get the pin number
  if (myParser->equalCmdParam(1,"P1") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P1, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF")) {
      digitalWrite(P1, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P2") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P2, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P2, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P3") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P3, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P3, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P4") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P4, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P4, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P5") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P5, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P5, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P6") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P6, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P6, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P7") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P7, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P7, LOW);
    }
  } else if (myParser->equalCmdParam(1,"P8") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P2, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(P8, LOW);
    }
  } else if (myParser->equalCmdParam(1,"PIN_FAN") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(PIN_FAN, HIGH);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(PIN_FAN, LOW);
    }
  } else if (myParser->equalCmdParam(1,"PIN_KEITHLEY") ) {
    if (myParser->equalCmdParam(2,"ON")) {
      digitalWrite(P2, LOW);
    } else if (myParser->equalCmdParam(2, "OFF") ) {
      digitalWrite(PIN_KEITHLEY, LOW);
    }
  }
}
