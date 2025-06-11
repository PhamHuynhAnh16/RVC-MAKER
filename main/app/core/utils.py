import os
import sys
import json
import codecs
import requests
import platform
import datetime

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning, gr_error
from main.app.variables import logger, translations, configs

def stop_pid(pid_file, model_name=None, train=False):
    try:
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join(configs["logs_path"], model_name, f"{pid_file}.txt")

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                os.kill(pid, 9)

            if os.path.exists(pid_file_path): os.remove(pid_file_path)

        pid_file_path = os.path.join(configs["logs_path"], model_name, "config.json")

        if train and os.path.exists(pid_file_path):
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            for pid in pids:
                os.kill(pid, 9)

            gr_info(translations["end_pid"])
    except:
        pass

def report_bug(error_info, provide):
    report_path = os.path.join(configs["logs_path"], "report_bugs.log")
    if os.path.exists(report_path): os.remove(report_path)

    report_url = codecs.decode(requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/jroubbx.gkg", "rot13")).text, "rot13")
    if not error_info: error_info = "Không Có"

    gr_info(translations["thank"])

    if provide:
        try:
            for log in [os.path.join(root, name) for root, _, files in os.walk(os.path.join(configs["logs_path"]), topdown=False) for name in files if name.endswith(".log")]:
                with open(log, "r", encoding="utf-8") as r:
                    with open(report_path, "a", encoding="utf-8") as w:
                        w.write(str(r.read()))
                        w.write("\n")
        except Exception as e:
            gr_error(translations["error_read_log"])
            logger.debug(e)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": f"Mô tả lỗi: {error_info}", "color": 15158332, "author": {"name": "Vietnamese_RVC", "icon_url": codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/vpb.cat", "rot13"), "url": codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP/gerr/znva","rot13")}, "thumbnail": {"url": codecs.decode("uggcf://p.grabe.pbz/7dADJbv-36fNNNNq/grabe.tvs", "rot13")}, "fields": [{"name": "Số Lượng Gỡ Lỗi", "value": content.count("DEBUG")}, {"name": "Số Lượng Thông Tin", "value": content.count("INFO")}, {"name": "Số Lượng Cảnh Báo", "value": content.count("WARNING")}, {"name": "Số Lượng Lỗi", "value": content.count("ERROR")}], "footer": {"text": f"Tên Máy: {platform.uname().node} - Hệ Điều Hành: {platform.system()}-{platform.version()}\nThời Gian Báo Cáo Lỗi: {datetime.datetime.now()}."}}]})

            with open(report_path, "rb") as f:
                requests.post(report_url, files={"file": f})
        except Exception as e:
            gr_error(translations["error_send"])
        finally:
            if os.path.exists(report_path): os.remove(report_path)
    else: requests.post(report_url, json={"embeds": [{"title": "Báo Cáo Lỗi", "description": error_info}]})

def google_translate(text, source='auto', target='vi'):
    if text == "": return gr_warning(translations["prompt_warning"])

    try:
        import textwrap

        def translate_chunk(chunk):
            response = requests.get(codecs.decode("uggcf://genafyngr.tbbtyrncvf.pbz/genafyngr_n/fvatyr", "rot13"), params={'client': 'gtx', 'sl': source, 'tl': target, 'dt': 't', 'q': chunk})
            return ''.join([i[0] for i in response.json()[0]]) if response.status_code == 200 else chunk

        translated_text = ''
        for chunk in textwrap.wrap(text, 5000, break_long_words=False, break_on_hyphens=False):
            translated_text += translate_chunk(chunk)

        return translated_text
    except:
        return text