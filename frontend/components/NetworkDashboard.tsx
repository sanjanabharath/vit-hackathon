import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Activity, Battery, Cpu, Thermometer, Users, Wifi } from "lucide-react";

const NetworkDashboard = () => {
  const [data, setData] = useState([]);
  const [currentMetrics, setCurrentMetrics] = useState({
    energyUsage: 0,
    networkLoad: 0,
    temperature: 0,
    activeUsers: 0,
    efficiency: 0,
  });

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const newDataPoint = {
        timestamp: new Date().toLocaleTimeString(),
        energyUsage: Math.random() * 100 + 50,
        networkLoad: Math.random() * 80 + 20,
        temperature: Math.random() * 15 + 20,
        activeUsers: Math.floor(Math.random() * 5000),
        efficiency: Math.random() * 20 + 80,
      };

      setCurrentMetrics(newDataPoint);
      //@ts-ignore
      setData((prevData) => [...prevData.slice(-20), newDataPoint]);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const MetricCard = ({ title, value, icon: Icon, unit, color }: any) => (
    <Card className="bg-white">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 text-${color}-500`} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {typeof value === "number" ? value.toFixed(1) : value} {unit}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="p-4 space-y-4 bg-gray-100">
      <h1 className="text-2xl font-bold mb-4">Network Energy Monitor</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Energy Usage"
          value={currentMetrics.energyUsage}
          icon={Battery}
          unit="kW"
          color="red"
        />
        <MetricCard
          title="Network Load"
          value={currentMetrics.networkLoad}
          icon={Activity}
          unit="%"
          color="blue"
        />
        <MetricCard
          title="Temperature"
          value={currentMetrics.temperature}
          icon={Thermometer}
          unit="°C"
          color="yellow"
        />
        <MetricCard
          title="Active Users"
          value={currentMetrics.activeUsers}
          icon={Users}
          unit=""
          color="green"
        />
        <MetricCard
          title="Efficiency"
          value={currentMetrics.efficiency}
          icon={Cpu}
          unit="%"
          color="purple"
        />
      </div>

      <Card className="bg-white p-4">
        <CardHeader>
          <CardTitle>Real-time Network Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <LineChart width={800} height={400} data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="energyUsage"
              stroke="#ef4444"
              name="Energy Usage (kW)"
            />
            <Line
              type="monotone"
              dataKey="networkLoad"
              stroke="#3b82f6"
              name="Network Load (%)"
            />
            <Line
              type="monotone"
              dataKey="temperature"
              stroke="#eab308"
              name="Temperature (°C)"
            />
          </LineChart>
        </CardContent>
      </Card>
    </div>
  );
};

export default NetworkDashboard;
