using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Flir.Atlas.Live;
using Flir.Atlas.Live.Discovery;
using Flir.Atlas.Live.Device;
using Flir.Atlas.Image;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Discovery discovery = new Discovery();

            // This will start the discovery process and wait for 'n' seconds
            List<CameraDeviceInfo> discoveredDevices = discovery.Start(5);

            // Check if any cameras are found
            if (discoveredDevices.Count > 0)
            {

               
                CameraDeviceInfo info = discoveredDevices[0];


                ThermalCamera cam = new ThermalCamera();
                cam.Connect(info);

                while (!cam.ConnectionStatus.Equals(ConnectionStatus.Connected)) ;
                while (cam.IsGrabbing != true) ;

                ImageBase image = cam.GetImage();
                image.SaveSnapshot("C:\\Users\\spsas\\Downloads\\img_check\\output_image.jpg");

                cam.Disconnect();
            }
            else
            {
                MessageBox.Show("No cameras found!");
            }

            discovery.Dispose();
        }
    }
}
