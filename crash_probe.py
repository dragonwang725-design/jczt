#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定向故障探针 v2 - 提取软件崩溃相关系统信息 + 完整硬件配置

如果显卡名称显示不完整，可以考虑安装 GPUtil 以获得更准确的采集：pip install GPUtil。届时程序会自动优先使用 GPUtil，不再依赖 wmic。
"""

import os
import re
import json
import psutil
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ========== 硬件信息采集 ==========
def get_hardware_info():
    """采集硬件配置信息"""
    info = {}
    # CPU
    info['cpu'] = {
        'model': platform.processor() or '未知',
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True)
    }
    # 如果能获取更详细的CPU信息（Windows）
    if platform.system() == 'Windows':
        try:
            cmd = 'wmic cpu get name'
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                info['cpu']['model'] = lines[1].strip()
        except:
            pass

    # 内存
    mem = psutil.virtual_memory()
    info['memory'] = {
        'total_gb': round(mem.total / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2)
    }

    # 显卡（优先用GPUtil，没有则用wmic）
    gpu_list = []
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_list.append({
                'name': gpu.name,
                'memory_total_mb': int(gpu.memoryTotal) if gpu.memoryTotal else '未知'
            })
    except ImportError:
        # 回退到 wmic 命令
        if platform.system() == 'Windows':
            try:
                cmd = 'wmic path win32_VideoController get name,AdapterRAM'
                result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:
                        if line.strip():
                            parts = line.split()
                            name = ' '.join(parts[:-1]) if len(parts) > 1 else parts[0]
                            # 显存单位转换（AdapterRAM 通常以字节为单位）
                            ram = parts[-1] if parts[-1].isdigit() else '未知'
                            if ram != '未知':
                                ram_mb = round(int(ram) / (1024**2), 1)
                                ram = f"{ram_mb} MB"
                            gpu_list.append({'name': name, 'memory_total': ram})
            except:
                pass
    info['gpu'] = gpu_list if gpu_list else [{'name': '未检测到独立显卡', 'memory_total': 'N/A'}]
    return info

# ========== 定向探针 ==========
def probe_crash(software_name: str):
    """针对指定软件，采集关联的系统信息 + 硬件配置"""
    report = {
        "probe_time": datetime.now().isoformat(),
        "software": software_name,
        "hardware": get_hardware_info(),   # 新增硬件部分
        "process_info": None,
        "event_logs": [],
        "system_resources": {},
        "crash_dumps": []
    }

    # 1. 检查进程状态
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']):
        if software_name.lower() in proc.info['name'].lower():
            report['process_info'] = {
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'cpu': proc.info['cpu_percent'],
                'memory': proc.info['memory_percent'],
                'started': datetime.fromtimestamp(proc.info['create_time']).isoformat()
            }
            break
    else:
        report['process_info'] = "进程未在运行"

    # 2. 系统资源快照
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/') if platform.system() != 'Windows' else psutil.disk_usage('C:')
    report['system_resources'] = {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory_used_percent': mem.percent,
        'memory_available_gb': round(mem.available / (1024**3), 2),
        'disk_free_gb': round(disk.free / (1024**3), 2)
    }

    # 3. Windows 事件日志（最近1小时）
    if platform.system() == 'Windows':
        try:
            safe_name = software_name.replace("'", "''")  # 转义单引号
        time_filter = (datetime.now() - timedelta(hours=1)).isoformat()
        cmd = f'powershell -Command "Get-WinEvent -FilterHashtable @{{LogName=\'Application\'; Level=2; StartTime=\'{time_filter}\'}} | Where-Object {{$_.Message -like \'*{safe_name}*\'}} | Select-Object -First 5 TimeCreated, Id, Message | ConvertTo-Json"'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.stdout.strip():
                logs = json.loads(result.stdout)
                if isinstance(logs, dict):
                    logs = [logs]
                report['event_logs'] = logs
        except Exception as e:
            report['event_logs'] = [{"error": str(e)}]

    # 4. 崩溃转储文件
    dump_paths = [
        Path(os.environ.get('LOCALAPPDATA', '')) / 'CrashDumps',
        Path.home() / 'AppData' / 'Local' / 'CrashDumps'
    ]
    for path in dump_paths:
        if path.exists():
            for dump in path.glob(f'*{software_name}*.dmp'):
                report['crash_dumps'].append({
                    'file': str(dump),
                    'size_mb': round(dump.stat().st_size / (1024**2), 2),
                    'modified': datetime.fromtimestamp(dump.stat().st_mtime).isoformat()
                })
    
    return report

# ========== 保存报告 ==========
def save_report(report, output_dir='.'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"故障诊断_{report['software']}_{ts}"
    json_path = Path(output_dir) / f"{base}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # 同时生成易读文本
    txt_path = Path(output_dir) / f"{base}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"故障诊断报告 - 软件: {report['software']}\n")
        f.write("="*60 + "\n\n")
        f.write(f"采集时间: {report['probe_time']}\n\n")

        f.write("【硬件配置】\n")
        hw = report['hardware']
        f.write(f"CPU: {hw['cpu']['model']} (物理{ hw['cpu']['physical_cores']}核 / 逻辑{hw['cpu']['logical_cores']}核)\n")
        f.write(f"内存: 总计 {hw['memory']['total_gb']} GB，可用 {hw['memory']['available_gb']} GB\n")
        for gpu in hw['gpu']:
            f.write(f"显卡: {gpu['name']} (显存 {gpu.get('memory_total', 'N/A')})\n")
        f.write("\n")

        f.write("【进程状态】\n")
        if isinstance(report['process_info'], str):
            f.write(f"  {report['process_info']}\n")
        else:
            p = report['process_info']
            f.write(f"  进程 PID: {p['pid']}, 名称: {p['name']}\n")
            f.write(f"  CPU占用: {p['cpu']}%, 内存占用: {p['memory']}%\n")
            f.write(f"  启动时间: {p['started']}\n")
        f.write("\n")

        f.write("【系统资源快照】\n")
        res = report['system_resources']
        f.write(f"  CPU使用率: {res['cpu_percent']}%\n")
        f.write(f"  内存使用率: {res['memory_used_percent']}%\n")
        f.write(f"  可用内存: {res['memory_available_gb']} GB\n")
        f.write(f"  磁盘剩余空间: {res['disk_free_gb']} GB\n\n")

        f.write("【最近事件日志（1小时内，含软件名）】\n")
        for log in report['event_logs']:
            if isinstance(log, dict):
                f.write(f"  时间: {log.get('TimeCreated', '')}\n")
                f.write(f"  事件ID: {log.get('Id', '')}\n")
                msg = log.get('Message', '')[:200]
                f.write(f"  消息: {msg}...\n\n")
            else:
                f.write(f"  {log}\n")

        f.write("【崩溃转储文件】\n")
        if report['crash_dumps']:
            for dmp in report['crash_dumps']:
                f.write(f"  文件: {dmp['file']}\n")
                f.write(f"  大小: {dmp['size_mb']} MB, 修改时间: {dmp['modified']}\n")
        else:
            f.write("  未找到崩溃转储文件。\n")

    print(f"✅ 报告已生成:\n  文本: {txt_path}\n  JSON: {json_path}")

if __name__ == "__main__":
    software = input("请输入要诊断的软件名称（如 chrome.exe）: ").strip()
    if not software:
        print("软件名称不能为空。")
        exit(1)
    report = probe_crash(software)
    save_report(report, output_dir='.')