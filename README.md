# Implementation of an autonomous vehicle using machine learning techniques and a low-power computer system
# Table of contents
- [Description](#description)
- [Autonomous Vehicle](#autonomous-vehicle)
- [Autonomous Driving System](#autonomous-driving-system)

# Description
- Developed an autonomous driving system based on Deep Learning methods implemented in Raspberry Pi 4 Model B 8GB. Built a vehicle using a 4WD car chassis kit, a webcam, an ultrasonic distance sensor, and other electronic components.
- A Convolutional Neural Network model takes raw input images and outputs the corresponding steering angles. The model is trained and evaluated using data collected by driving the vehicle in a test track under various conditions.
- The vehicle uses a distance sensor to react immediately to unanticipated obstacles by automatically controlling its speed.
- Experimental results show that the vehicle can autonomously navigate on the test track.

[![Youtube Demo of Car](https://user-images.githubusercontent.com/98546890/154834752-57ca3f17-5531-4e63-999d-8e70681edba1.png)](https://youtu.be/Ffq6Z1WhVpw)

# Autonomous Vehicle
## Car Parts
- Raspberry Pi 4 Model B 8GB
- Aluminum Case with Fan
- SanDisk Extreme PRO microSD 64GB
- Aigoss Webcam 1080P Full HD
- Distance Sensor SRF05
- 4WD Robot Car Chassis Kit
- Motor Driver Controller Board L298N
- Power Bank AUKEY PB-Y23
- USB C Cable
- Battery 18650 (Quantity: 2)
- Battery Holder / Case for 2x 18650 7.4V
- Switch 6A/250V ON/OFF

## Prototyped Autonomous Vehicle
![front-right](https://user-images.githubusercontent.com/98546890/154454520-ca721fa5-c6bd-47fa-9d01-fb2a353518c0.jpg)
![back-left](https://user-images.githubusercontent.com/98546890/154454464-2703b6ea-b829-4793-8c9b-e128c75b8f62.jpg)
![under](https://user-images.githubusercontent.com/98546890/154454549-769360b2-82f5-487c-a0b7-9f960f678d8e.jpg)

# Autonomous Driving System
### The development phases to build the Autonomous Driving system:
1. In the beginning, the images used to train the neural network were collected. A total of 2,204 images were collected for this purpose.
2. A Convolutional Neural Network model is trained and evaluated using the collected data. Results of model training:

<table>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154844861-7669e51c-93d2-43bc-93cc-085d6ebc761d.png"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154844866-a95b251d-3157-4384-a32c-f6d2eb9ad215.png"></td>
      </tr>
</table>

3. The Autonomous Driving pipeline steps:
   1. Pre-processing images
   2. Model takes input images and outputs the corresponding steering angles
   3. Robotic car turns accordingly to model output values

### Example of Autonomous Driving
Top-left corner shows the output model value in yellow. Top-right corner shows the image index in green.
<table>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845106-42043c76-499b-47a2-a007-e6306fb9216e.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845119-5de7acba-33b9-43af-b954-de0436b296ff.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845122-fdf621ff-9b58-48fc-8c36-8a16fdacd9d5.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845123-68a68449-bcc7-4c3a-a822-ef8a68ae321f.jpg"></td>
      </tr>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845124-05255906-e8f8-4935-b7c2-fce899baffeb.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845125-2f8e8374-3e41-46e1-b908-e2c803365a2d.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845127-bede73a8-ca8c-4223-b213-b9497fff3011.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845128-8cf8b927-e683-43a8-b061-bc4e36adb7c5.jpg"></td>
      </tr>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845131-821c55a5-f765-4d4e-aac3-769739ea3f63.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845133-57558132-6719-48bd-8d87-45eeb2f9e35f.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845134-aa8aeb55-149e-4a9d-b33c-fe69e92597b6.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845135-ca7abd3b-9bf6-438a-ad36-ebf9d6d1f7ce.jpg"></td>
      </tr>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845137-8ae723f8-b246-4cef-82fd-96656296bede.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845138-287b1ecd-4262-48b1-a15b-1db039be079f.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845139-0141212d-d09c-4bd8-8e92-c44a48f68a39.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845140-5a431600-9fa7-4e37-8d12-6b608345a7fb.jpg"></td>
      </tr>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845141-85c3cf19-3c78-4202-80f2-e7d471d921e7.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845142-75533eae-25e7-45f8-be0f-a10c2781eb50.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845143-2caf1cf0-a841-4de4-af95-81a9fa73a7f5.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845144-993f4aa4-64cc-42fa-acc5-35d193db0f2f.jpg"></td>
      </tr>
      <tr>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845145-78f536f2-6b9d-4b0b-90c6-e0be5b8b2b8d.jpg"></td>
            <td><img src="https://user-images.githubusercontent.com/98546890/154845147-201ec31a-74fe-4767-abd6-638394cdccfb.jpg"></td>
      </tr>
</table>
