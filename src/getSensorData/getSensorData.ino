/*!
 * @file getSensorData.ino
 * @brief Get the sensor's gyroscope, accelerometer and onboard temperature.
 * @n connected table in SPI
 * -----------------------------------------------------------------------------------------------------
 *  sensor pin  |            MCU                    | ESP32 | ESP8266 |    M0   | micro:bit | Mega2560 |
 *    FSY       | not connected, floating           |   X   |    X    |    X    |     X     |     X    |
 *    INT       | not connected, floating           |   X   |    X    |    X    |     X     |     X    |
 *    CS        | connected to the IO pin of MCU    | 5/D8  |  5/D6   |    5    |     P8    |     5    |
 *    SDO       | connected to miso of mcu'spi      |19/MISO|  MISO   |  MISO   |  P14/MISO |  50/MISO |
 *    SDI       | connected to mosi of mcu'spi      |23/MOSI|  MOSI   |  MOSI   |  P15/MOSI |  51/MOSI |
 *    SCK       | connected to sck of mcu'spi       |18/SCK |  SCK    |  SCK    |  P13/SCK  |  52/SCK  |
 *    GND       | GND                               |  GND  |   GND   |   GND   |    GND    |    GND   |
 *    3V3/VCC   | 3V3/VCC                           |  3V3  |   3V3   |   3V3   |    3V3    |    5V    |
 * -----------------------------------------------------------------------------------------------------
 *
 * @n connected table in IIC
 * ---------------------------------------------------------------------------------------------------
 * sensor pin |            MCU                    | ESP32 | ESP8266 |    M0   | micro:bit | Mega2560 |
 *    FSY     | not connected, floating           |   X   |    X    |    X    |     X     |     X    |
 *    INT     | not connected, floating           |   X   |    X    |    X    |     X     |     X    |
 *    SDA     | connected to SDA of mcu'iic       | 21/SDA|   SDA   |   SDA   |  P20/SDA  |  20/SDA  |
 *    SCL     | connected to scl of mcu'iic       | 22/SCL|   SCL   |   SCL   |  P19/SCL  |  21/SCL  |
 *    GND     | GND                               |  GND  |   GND   |   GND   |    GND    |    GND   |
 *    3V3/VCC | 3V3/VCC                           |  3V3  |   3V3   |   3V3   |    3V3    |    5V    |
 * ---------------------------------------------------------------------------------------------------
 *
 * @copyright   Copyright (c) 2010 DFRobot Co.Ltd (http://www.dfrobot.com)
 * @license     The MIT License (MIT)
 * @author [Arya](xue.peng@dfrobot.com)
 * @version  V1.0
 * @date  2021-05-24
 * @url https://github.com/DFRobot/DFRobot_ICG20660L
 */
#include "DFRobot_ICG20660L.h"

#ifdef ARDUINO_BBC_MICROBIT
#define CS_PIN      8                      //The CS pin of sensor which is connected to the 8 digital io pin of micro:bit,and also can connected to other pin.
#else
#define CS_PIN      5                      //The CS pin of sensor which is connected to the 5 digital io pin of MCU,and also can connected to other pin.
#endif

const int buttonPin = 2;  // the number of the pushbutton pin
int buttonState = 0;  // variable for reading the pushbutton status
/**
 * @fn DFRobot_ICG20660L_IIC
 * @brief The constructor of the ICG20660L sensor using IIC communication.
 * @param addr:  7-bit IIC address, controlled by SDO pin.
 * @n     IIC_ADDR_SDO_H or 0x69:  SDO pull high.(default)
 * @n     IIC_ADDR_SDO_L or 0x68:  SDO pull down.
 * @param pWire:   TwoWire class pointer.
 */
DFRobot_ICG20660L_IIC icg(/*addr=*/IIC_ADDR_SDO_H, &Wire);
/**
 * @fn DFRobot_ICG20660L_SPI
 * @brief The constructor of the ICG20660L sensor, using SPI communication.
 * @param csPin:  SPI chip select pin, connected to IO pin of MCU.
 * @param spi: SPIClass class pointer.
 */
//DFRobot_ICG20660L_SPI icg(/*csPin=*/CS_PIN, &SPI);
void setup() {
  pinMode(buttonPin, INPUT);
  Serial.begin(115200);
  while(!Serial){                                                     //Waiting for USB Serial COM port to open.
  }
  
  Serial.print("Initialization sensor...");
  /**
   * @fn begin
   * @brief Initialize the sensor, after initialization, all sensors are turned off, and the corresponding configuration
   * @n needs to be turned on through enableSensor.
   * @param mode: Enum variable,from eDataReadMode_t, configure to read sensor data from FIFO or register?
   * @n     eRegMode: Read sensor data from registers.
   * @n     eFIFOMode:Read sensor data from 512 bytes FIFO. 
   * @note  Read from FIFO, accelerometer, gyroscope and temperature must all be enabled, 
   * @n and the internal sampling rate must be configured to be consistent. 
   * @return status:
   * @n      0 :   Initialization success.
   * @n      -1:   Interface initialization failed(IIC or SPI).
   * @n      -2:   Failed to read the device ID, the ID is not 0x91
   */
  while(icg.begin(/*mode=*/icg.eRegMode) != 0){
      Serial.println("failed. Please check whether the hardware connection is wrong.");
      delay(1000);
      Serial.print("Initialization sensor...");
  }
  Serial.println("done.");
  
  Serial.print("ICG20660L Device ID: 0x");
  Serial.println(icg.readID(), HEX);
  
  /**
   * @fn enableSensor
   * @brief Enable sensor, including Accel of xyz axis, Gyro of xyz, temperature. 
   * @param bit: 8-bit byte data. Each bit represents enabling a function bit, as shown in the following table:
   * @n -------------------------------------------------------------------------------------------------------------------
   * @n |       bit7      |     bit6     |      bit5   |    bit4     |     bit3    |     bit2   |    bit1    |    bit0    |
   * @n -------------------------------------------------------------------------------------------------------------------
   * @n |     reserve     |    reserve   | eAccelAxisX | eAccelAxisY | eAccelAxisZ | eGyroAxisX | eGyroAxisY | eGyroAxisZ |
   * @n |                                |            eAccelAxisXYZ                |           eGyroAxisXYZ               |
   * @n |                                |                                eAxisAll                                        |
   * @n -------------------------------------------------------------------------------------------------------------------
   * @n   bit0:  Z-axis of gyro and temperature.
   * @n   bit1:  Y-axis of gyro and temperature.
   * @n   bit2:  X-axis of gyro and temperature.
   * @n   bit3:  Z-axis of acceleration.
   * @n   bit4:  Z-axis of acceleration.
   * @n   bit5:  Z-axis of acceleration.
   * @n   bit6:  reserve.
   * @n   bit7:  reserve.
   * @note Enabling any axis of the gyroscope will automatically enable the on-board temperature sensor.
   * @n   eGyroAxisZ: The bit0 of the bit, enable gyro's z axis and temperature.
   * @n   eGyroAxisY: The bit1 of the bit, enable gyro's y axis and temperature.
   * @n   eGyroAxisX: The bit2 of the bit, enable gyro's X axis and temperature.
   * @n   eAccelAxisZ: The bit3 of the bit, enable accel's z axis.
   * @n   eAccelAxisY: The bit4 of the bit, enable Accel's y axis.
   * @n   eAccelAxisX: The bit5 of the bit, enable Accel's X axis.
   * @n   eGyroAxisXYZ or eGyroAxisX|eGyroAxisY|eGyroAxisZ: The bit0/bit1/bit2 of the bit, enable gyro's xyz axis and temperature.
   * @n   eAccelAxisXYZ or eAccelAxisX|eAccelAxisY|eAccelAxisZ: The bit3/bit4/bit5 of the bit, enable Accel's xyz axis.
   * @n   eAxisAll or eGyroAxisX|eGyroAxisY|eGyroAxisZ|eAccelAxisX|eAccelAxisY|eAccelAxisZ: The bit0/bit1/bit2/bit3/bit4/bit5 of the bit,
   * @n enable temperature, Accel's and gyro's xyz axis. 
   */
  icg.enableSensor(icg.eAxisAll);
  //icg.enableSensor(icg.eGyroAxisXYZ|icg.eAccelAxisXYZ);
  //icg.enableSensor(icg.eGyroAxisX|icg.eGyroAxisY|icg.eGyroAxisZ|icg.eAccelAxisX|icg.eAccelAxisY|icg.eAccelAxisZ);
  /**
   * @fn configGyro
   * @brief Config of gyro's full scale, dlpf bandwidth and internal sample rate. 
   * @param scale  The full scale of gyro, unit: dps(Degrees per second).
   * @n     eFSR_G_125DPS:  The full scale range is ±125 dps.
   * @n     eFSR_G_250DPS:  The full scale range is ±250 dps.
   * @n     eFSR_G_500DPS:  The full scale range is ±500 dps.
   * @param bd  Set 3-db bandwidth.
   * @n     eGyro_DLPF_8173_32KHZ: When the signal is equal to or greater than 8173Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 32KHz.
   * @n     eGyro_DLPF_3281_32KHZ: When the signal is equal to or greater than 3281Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 32KHz.
   * @n     eGyro_DLPF_250_8KHZ:   When the signal is equal to or greater than 250Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 8KHz.
   * @n     eGyro_DLPF_176_1KHZ:   When the signal is equal to or greater than 176Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eGyro_DLPF_92_1KHZ:    When the signal is equal to or greater than 92Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eGyro_DLPF_3281_8KHZ:  When the signal is equal to or greater than 3281Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 8KHz.
   * @note  When the gyroscope and accelerometer are both enabled, if the sensor data is read through the FIFO,
   * @n the internal sampling rate of the gyroscope and accelerometer must be the same.
   */
  icg.configGyro(/*scale=*/icg.eFSR_G_125DPS, /*bd=*/icg.eGyro_DLPF_176_1KHZ);
  /**
   * @fn configAccel
   * @brief Config of accel's full scale, dlpf bandwidth and internal sample rate. 
   * @param scale  The full scale of accel, unit: g(1g = 9.80665 m/s²).
   * @n     eFSR_A_2G:  The full scale range is ±2g.
   * @n     eFSR_A_4G:  The full scale range is ±4g.
   * @n     eFSR_A_8G:  The full scale range is ±8g.
   * @n     eFSR_A_16G:  The full scale range is ±16g.
   * @param bd  Set 3-db bandwidth.
   * @n     eAccel_DLPF_5_1KHZ or 0:    When the signal is less than or equal to 5Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eAccel_DLPF_10_1KHZ or 1:   When the signal is less than or equal to 10Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eAccel_DLPF_21_1KHZ or 2:   When the signal is less than or equal to 21Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eAccel_DLPF_44_1KHZ or 3:   When the signal is less than or equal to 44Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eAccel_DLPF_99_1KHZ or 4:   When the signal is less than or equal to 99Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.
   * @n     eAccel_DLPF_218_1KHZ or 5:  When the signal is less than or equal to 218Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.    Support low power consumption mode
   * @n     eAccel_DLPF_420_1KHZ or 6:  When the signal is less than or equal to 420Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.   Support low power consumption mode
   * @n     eAccel_DLPF_1046_4KHZ or 7: When the signal is less than or equal to 1046Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 4KHz.   Support low power consumption mode
   * @n     eAccel_DLPF_55_1KHZ or 8:   When the signal is less than or equal to 55Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.   Only support low power consumption mode
   * @n     eAccel_DLPF_110_1KHZ or 9:  When the signal is less than or equal to 110Hz, there will be obvious attenuation, 3-db attenuation, and the internal sampling rate is 1KHz.    Only support low power consumption mode
   * @note When the gyroscope and accelerometer are both enabled, if the sensor data is read through the FIFO, 
   * @n the internal sampling rate of the gyroscope and accelerometer must be the same.
   * @param odr:  Set the frequency of waking up the chip to take a sample of accel data – the low power accel Output Data Rate.
   * @n     eODR_125Hz or 9:    The low power accel Output Data Rate: 125Hz
   * @n     eODR_250Hz or 10:   The low power accel Output Data Rate: 250Hz
   * @n     eODR_500Hz or 11:   The low power accel Output Data Rate: 500Hz
   * @param lowPowerFlag:  Whether to configure the Acceleration to low power mode.
   * @n     true:          Enter low power mode.
   * @n     false:         Not configure the Acceleration to low power mode.(default)
   */
  icg.configAccel(/*scale=*/icg.eFSR_A_2G, /*bd=*/icg.eAccel_DLPF_218_1KHZ);
  /**
   * @fn setSampleDiv
   * @brief Set sample rate divider. 
   * @param div  Sample rate divider, the range is 0~255.
   * @n     Sampling rate = internal sampling rate/(div+1)
   * @note  If the accelerometer configuration is in low power consumption mode, that is, the formal parameter lowPowerFlag of the configAccel function is true, 
   * @n the sampling rate must match the output rate of the formal parameter odr of configAccel, as shown in the following table:
   * @n ----------------------------------------------------------------------------
   * @n |                           configAccel                    |  setSampleDiv  |
   * @n ----------------------------------------------------------------------------|
   * @n |            bd             |      odr      | lowPowerFlag |      div       |
   * @n ----------------------------------------------------------------------------|
   * @n |            X              |       X       |    false     |      0~255     |
   * @n ----------------------------------------------------------------------------|
   * @n |                           |  eODR_125Hz   |    true      |        7       |
   * @n |                           |-----------------------------------------------|
   * @n |bd of supporting low power |  eODR_250Hz   |    true      |        3       |
   * @n |consumption mode           |-----------------------------------------------|
   * @n |                           |  eODR_500Hz   |    true      |        1       |
   * @n |---------------------------------------------------------------------------|
   */
  icg.setSampleDiv(0);
}

#define printAxisData(sAxis, str) \
  Serial.print(sAxis.x); \
  Serial.print(str); \
  Serial.print(sAxis.y); \
  Serial.print(str); \
  Serial.print(sAxis.z);\

void loop() {
  sIcg20660SensorData_t gyro, accel;
  float t;
  buttonState = digitalRead(buttonPin);
  /**
   * @fn getSensorData
   * @brief Get Sensor's accel, gyro and temperature data.
   * @param accel: sIcg20660SensorData_t structure pointer which points to accel or NULL.
   * @param gyro: sIcg20660SensorData_t structure pointer which points to gyro or NULL.
   * @param t:  A float pointer which points to temperature or NULL.
   */
  if (buttonState == LOW) {
    icg.getSensorData(&accel, &gyro, &t);
    // printAxisData(accel, ",");
    // Serial.println();
    printAxisData(gyro, ",");
    Serial.println();
  }
}


