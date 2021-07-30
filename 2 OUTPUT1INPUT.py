import cv2
import RPi.GPIO as GPIO
import time

# Pin Definitons:
led_pin = 13  #OUTPUT1
led_pin1 = 19  # OUTPUT 2 
but_pin = 15  # INPUT PIN
dispH=960
dispW=1280
flip=2

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(led_pin, GPIO.OUT) 
    GPIO.setup(led_pin1, GPIO.OUT) # LED pin set as output
    GPIO.setup(but_pin, GPIO.IN)  # button pin set as input

    # Initial state for LEDs:
    GPIO.output(led_pin, GPIO.LOW)
    GPIO.output(led_pin1, GPIO.LOW)

    print("Starting demo now! Press CTRL+C to exit")
    i=1
    while i<100:
    	print("Waiting for INPUT event FROM PIN")
    	x = GPIO.input(but_pin)
    	print(x)
    	if x==1:
    		print("Button Pressed!",i)
    		i+=1
    		GPIO.output(led_pin, GPIO.LOW)
			time.sleep(2)
    		GPIO.output(led_pin, GPIO.HIGH)
    		time.sleep(2)
    		GPIO.cleanup()
    		print("Sumit")
    		GPIO.output(led_pin, GPIO.LOW)
			time.sleep(2)
    		GPIO.output(led_pin, GPIO.HIGH)
    		time.sleep(2)
    		GPIO.cleanup()
    		print("Sumit")

if __name__ == '__main__':
          main()
