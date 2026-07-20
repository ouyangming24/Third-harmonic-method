import time
import zhinst.core
import numpy as np
import pandas as pd


class ZurichLockInControl:
    def __init__(self, device_id, server_host='192.168.59.62', server_port=8004, api_level=None):
        # 初始化设备
        self.device_id = device_id

        # 尝试使用指定 API 级别连接，如果失败（例如 HF2LI 仅支持 api_level=1），则降级到 API 级别 1
        # HF2LI 通常使用 8005 端口，若是 8005，默认使用 api_level=1 避免 20 秒超时
        if api_level is None:
            api_level = 1 if server_port == 8005 else 6

        try:
            self.daq = zhinst.core.ziDAQServer(server_host, server_port, api_level)
        except RuntimeError as e:
            if api_level != 1:
                print(f"API level {api_level} connection failed: {e}. Retrying with API level 1 (HF2LI mode)...")
                self.daq = zhinst.core.ziDAQServer(server_host, server_port, 1)
            elif api_level == 1 and server_port == 8005:
                print(f"API level 1 failed: {e}. Retrying with API level 6...")
                self.daq = zhinst.core.ziDAQServer(server_host, server_port, 6)
            else:
                raise
        print(f"Connected to device: {device_id}")

    def configure_signal_inputs(self, input_channel=0, range=1.0, offset=0.0, ac=False, diff=False):
        """
        配置 Signal Inputs 参数
        """
        self.daq.setInt(f'/{self.device_id}/sigins/{input_channel}/ac', int(ac))  # AC/DC 耦合
        self.daq.setInt(f'/{self.device_id}/sigins/{input_channel}/diff', int(diff))  # 差分/单端模式
        self.daq.setDouble(f'/{self.device_id}/sigins/{input_channel}/range', range)  # 信号范围
        print(f"Signal Inputs Configured: channel={input_channel}, range={range}, "
              f"offset={offset}, ac={ac}, diff={diff}")

    def configure_lockin(self, freq, order_time, time_constant):
        # 设置锁相放大器的输出幅值、频率和时间常数
        self.daq.setDouble(f'/{self.device_id}/oscs/0/freq', freq)  # 使用振荡器0作为基频

        # 配置解调器0，锁定1次谐波
        self.daq.setInt(f'/{self.device_id}/demods/0/oscselect', 0)
        self.daq.setInt(f'/{self.device_id}/demods/0/harmonic', 1)
        self.daq.setDouble(f'/{self.device_id}/demods/0/order', order_time)
        self.daq.setDouble(f'/{self.device_id}/demods/0/timeconstant', time_constant)

        # 配置解调器1，锁定3次谐波
        self.daq.setInt(f'/{self.device_id}/demods/1/oscselect', 0)
        self.daq.setInt(f'/{self.device_id}/demods/1/harmonic', 3)
        self.daq.setDouble(f'/{self.device_id}/demods/1/order', order_time)
        self.daq.setDouble(f'/{self.device_id}/demods/1/timeconstant', time_constant)

        print(f"Configured lock-in: base_freq={freq}, "
              f"time_constant={time_constant}, demodulators for 1f and 3f.")

        # 关闭所有解调器的数据传输
        for demod in range(4):
            self.daq.setInt(f'/{self.device_id}/demods/{demod}/enable', 0)
        # 启用指定解调器的数据传输
        self.daq.setInt(f'/{self.device_id}/demods/0/enable', 1)  # 启用1f
        self.daq.setInt(f'/{self.device_id}/demods/1/enable', 1)  # 启用3f

    def configure_signal_outputs(self, output_channel=0, amplitude=1.5, on=True, range_val=10.0, offset=0.0):
        """
        配置 Signal Outputs 参数
        """
        self.daq.setInt(f'/{self.device_id}/sigouts/{output_channel}/on', int(on))  # 开启或关闭输出

        # 使用振荡器0作为信号输出源
        self.daq.setInt(f'/{self.device_id}/sigouts/{output_channel}/enables/0', 1)
        self.daq.setInt(f'/{self.device_id}/sigouts/{output_channel}/enables/1', 0)
        self.daq.setInt(f'/{self.device_id}/sigouts/{output_channel}/enables/2', 0)
        self.daq.setInt(f'/{self.device_id}/sigouts/{output_channel}/enables/3', 0)

        # 苏黎世仪器的 range 只能取几个特定值（比如 0.01, 0.1, 1.0, 10.0）。
        # 如果 range 设置过小，下发的 amplitude 就会被强行截断到该 range 的最大值
        ranges = [0.01, 0.1, 1.0, 10.0]
        actual_range = 10.0
        for r in ranges:
            if r >= amplitude:
                actual_range = r
                break

        # 很多 Zurich 型号（包括 HF2LI）在量程不够时设置 amplitude 会被截断，
        # 我们必须确保下发的范围大于等于预期的振幅，并保证其先生效
        self.daq.setDouble(f'/{self.device_id}/sigouts/{output_channel}/range', actual_range)
        self.daq.sync()  # 同步确保 range 生效

        # 区分设备类型：HF2LI 的 amplitude 是一个无量纲的比例 (0 到 1.0)，实际输出电压 = range * amplitude
        # 而其他新型号（如 MFLI, UHFLI）直接输入电压值 (V)
        device_type = ""
        try:
            device_type = self.daq.getString(f'/{self.device_id}/features/devtype')
        except Exception:
            pass

        if "HF2" in device_type:
            amp_value = amplitude / actual_range
        else:
            amp_value = amplitude

        self.daq.setDouble(f'/{self.device_id}/sigouts/{output_channel}/amplitudes/0', amp_value)  # 设置振荡器0的幅值
        self.daq.setDouble(f'/{self.device_id}/sigouts/{output_channel}/offset', offset)  # 输出偏置
        self.daq.sync()
        print(
            f"Signal Outputs Configured: channel={output_channel}, on={on}, amplitude={amplitude} (raw={amp_value}), range={actual_range}, offset={offset}")

    def read_demod_data(self, duration=1.0, sample_rate=1e3):
        # 订阅解调器的信号
        path1 = f'/{self.device_id}/demods/0/sample'  # 1f
        path3 = f'/{self.device_id}/demods/1/sample'  # 3f
        self.daq.subscribe([path1, path3])
        self.daq.sync()

        # 等待数据积累
        print("Acquiring demodulator data for 1f and 3f...")
        time.sleep(duration)

        # 调用 poll 方法获取数据
        data = self.daq.poll(recording_time_s=duration, timeout_ms=500, flat=True)
        self.daq.unsubscribe('*')

        # 检查数据是否存在
        x1, y1, x3, y3 = [], [], [], []
        if path1 in data:
            samples1 = data[path1]
            x1 = samples1['x']
            y1 = samples1['y']
        else:
            print("No data received for 1f demodulator.")

        if path3 in data:
            samples3 = data[path3]
            x3 = samples3['x']
            y3 = samples3['y']
        else:
            print("No data received for 3f demodulator.")

        return x1, y1, x3, y3

    def calculate_amplitude_phase(self, x, y):
        if len(x) == 0 or len(y) == 0:
            print("No valid data for amplitude and phase calculation.")
            return None, None, None, None

        # 计算幅值和相位
        x_mean = np.mean(np.array(x))
        y_mean = np.mean(np.array(y))
        # 幅值计算基于X和Y的平均值，这与相位计算一致
        amplitude = np.sqrt(x_mean ** 2 + y_mean ** 2)
        phase = np.arctan2(y_mean, x_mean)
        phase_deg = np.degrees(phase)  # 转换为角度
        return amplitude, phase_deg, x_mean, y_mean

    def save_results_to_excel(self, frequency, output_voltage, amplitude1, phase1, amplitude3, phase3, x3_mean, y3_mean,
                              filename="byd-YJ1-100-1-829-电压源430mv.xlsx"):
        # 如果文件不存在，创建新的 DataFrame 并写入 Excel
        amplitude1 = round(amplitude1, 9)
        amplitude3 = round(amplitude3, 9)
        x3_mean = round(x3_mean, 9)
        y3_mean = round(y3_mean, 9)
        try:
            df = pd.read_excel(filename)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Frequency", "V3_real", "V3_imag", "Output Voltage", "Amplitude_1f", "Phase_1f",
                                       "Amplitude_3f", "Phase_3f"])

        # 将新数据追加到 DataFrame 中
        new_data = pd.DataFrame({
            "Frequency": [frequency],
            "V3_real": [x3_mean],
            "V3_imag": [y3_mean],
            "Output Voltage": [output_voltage],
            "Amplitude_1f": [amplitude1],
            "Phase_1f": [phase1],
            "Amplitude_3f": [amplitude3],
            "Phase_3f": [phase3]

        })
        df = pd.concat([df, new_data], ignore_index=True)

        # 保存到 Excel 文件
        df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")

    def disconnect(self):
        # 断开连接
        self.daq.disconnect()
        print("Disconnected from the device.")

    def wait_for_integration(self, time_constant, duration_multiplier=5):
        """
        等待积分时间，确保信号经过足够时间的积分，确保数据稳定。
        :param time_constant: 锁相放大器的时间常数
        :param duration_multiplier: 等待时间的倍数，默认是时间常数的 15 倍
        """
        wait_time = time_constant * duration_multiplier
        print(f"Waiting for integration to stabilize. Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time)  # 等待足够的时间，使数据稳定


# 主程序
if __name__ == "__main__":
    # 替换为您的设备 ID，例如 'dev3791'
    DEVICE_ID = 'dev3791'

    lockin = ZurichLockInControl(DEVICE_ID)
    freqs = np.linspace(100, 1, 30)  # 生成频率范围从600到100的10个频率点

    output_amplitude = 7.8  # 输出电压幅值

    try:
        # 配置锁相放大器
        lockin.configure_signal_inputs(
            input_channel=0,
            range=0.1,  # 输入信号范围
            ac=False,  # 启用 AC 耦合
            diff=True  # 单端模式
        )

        lockin.configure_signal_outputs(
            output_channel=0,
            amplitude=output_amplitude,
            on=True,
            range_val=10.0,
            offset=0.0
        )
        for freq in freqs:
            # 计算对应的时间常数（根据频率）
            time_constant = 20 / freq

            # 配置锁相放大器频率和时间常数
            lockin.configure_lockin(
                freq=freq,
                order_time=4,
                time_constant=time_constant
            )

            # 等待积分时间结束，确保数据稳定
            lockin.wait_for_integration(time_constant=time_constant)

            # 读取解调数据并计算幅值和相位
            x1, y1, x3, y3 = lockin.read_demod_data(duration=2)
            amplitude1, phase1, _, _ = lockin.calculate_amplitude_phase(x1, y1)
            amplitude3, phase3, x3_mean, y3_mean = lockin.calculate_amplitude_phase(x3, y3)

            # 保存结果到 Excel 表格
            if all(v is not None for v in [amplitude1, phase1, amplitude3, phase3, x3_mean, y3_mean]):
                lockin.save_results_to_excel(freq, output_amplitude, amplitude1, phase1, amplitude3, phase3, x3_mean,
                                             y3_mean)

    finally:
        # 确保资源释放
        lockin.disconnect()
