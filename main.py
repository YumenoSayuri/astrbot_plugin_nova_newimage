import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider import Provider


@register(
    "astrbot_plugin_newimage",
    "辉宝",
    "AI生图插件：支持图生图(手办化/Q版化等预设)、文生图、自定义Prompt，含次数限制与签到系统",
    "1.1.0",
    "https://github.com/huibao/astrbot_plugin_newimage",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url: logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在尝试下载图片: {url}")
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except aiohttp.ClientResponseError as e:
                logger.error(f"图片下载失败: HTTP状态码 {e.status}, URL: {url}, 原因: {e.message}")
                return None
            except asyncio.TimeoutError:
                logger.error(f"图片下载失败: 请求超时 (30s), URL: {url}")
                return None
            except Exception as e:
                logger.error(f"图片下载失败: 发生未知错误, URL: {url}, 错误类型: {type(e).__name__}, 错误: {e}",
                             exc_info=True)
                return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit(): logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。"); return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e:
                logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw: return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def _image_component_to_bytes(self, image_comp: Image) -> bytes | None:
            """
            将 Image 组件转换为字节。
            优先使用 AstrBot 提供的 convert_to_base64，兼容 WebChat 等平台的图片来源。
            """
            # 1. 直接尝试 convert_to_base64
            if hasattr(image_comp, "convert_to_base64"):
                try:
                    base64_str = await image_comp.convert_to_base64()
                    if base64_str:
                        if base64_str.startswith("data:image/"):
                            base64_str = base64_str.split(",", 1)[1]
                        return base64.b64decode(base64_str)
                except Exception as e:
                    logger.warning(f"通过 convert_to_base64 获取图片数据失败: {e}", exc_info=True)

            # 2. 回退到原有逻辑
            if image_comp.url:
                return await self._load_bytes(image_comp.url) or None
            if image_comp.file:
                return await self._load_bytes(image_comp.file) or None
            return None

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if img := await self._image_component_to_bytes(s_chain):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if img := await self._image_component_to_bytes(seg):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            if avatar := await self._get_avatar(event.get_sender_id()):
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None
        self.group_task_counts: Dict[str, int] = {}
        self.queue_lock = asyncio.Lock()
        self.group_task_limit: int = 0
        # 供应商相关
        self.provider_id: str = ""
        self.provider: Optional[Provider] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        limit_raw = self.conf.get("group_task_limit", 2)
        try:
            self.group_task_limit = max(0, int(limit_raw))
        except (TypeError, ValueError):
            self.group_task_limit = 0
            logger.warning(f"NewImage: group_task_limit 配置无效 ({limit_raw})，已按 0 处理")
        self.group_task_counts.clear()
        
        # 加载供应商配置
        self.provider_id = self.conf.get("provider_id", "")
        if self.provider_id:
            self.provider = self.context.get_provider_by_id(self.provider_id)
            if self.provider:
                logger.info(f"NewImage 插件已加载，使用提供商: {self.provider_id}")
            else:
                logger.warning(f"NewImage: 未找到提供商 '{self.provider_id}'，将使用手动配置")
        else:
            logger.info("NewImage 插件已加载，使用手动API配置")
            if not self.conf.get("api_keys") and not self.conf.get("api_url"):
                logger.warning("NewImage: 未配置提供商，也未配置手动API，插件可能无法工作")

    async def _load_prompt_map(self):
        logger.info("正在加载 prompts...")
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            try:
                if ":" in item:
                    key, value = item.split(":", 1)
                    self.prompt_map[key.strip()] = value.strip()
                else:
                    logger.warning(f"跳过格式错误的 prompt (缺少冒号): {item}")
            except ValueError:
                logger.warning(f"跳过格式错误的 prompt: {item}")
        logger.info(f"加载了 {len(self.prompt_map)} 个 prompts。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
        text = event.message_str.strip()
        if not text: return
        cmd = text.split()[0].strip()
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False
        if cmd == bnn_command:
            user_prompt = text.removeprefix(cmd).strip()
            is_bnn = True
            if not user_prompt: return
        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)
        else:
            return
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            
            # 修复次数限制逻辑：
            # 1. 如果开启了用户限制且用户次数为0，检查是否有群组次数可用
            # 2. 如果开启了群组限制且群组次数为0，检查是否有用户次数可用
            # 3. 只有当所有启用的限制都没有可用次数时才拒绝
            can_use_user_count = not user_limit_on or user_count > 0
            can_use_group_count = not group_limit_on or group_count > 0
            
            # 核心逻辑：必须至少有一个可用的次数来源
            if group_id:
                # 在群聊中：如果两种限制都开启，需要至少一种有次数
                if user_limit_on and group_limit_on:
                    if user_count <= 0 and group_count <= 0:
                        yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。\n请使用「辉宝赐福」获取次数。")
                        return
                elif user_limit_on and user_count <= 0:
                    # 只开启用户限制，用户次数为0
                    yield event.plain_result("❌ 您的使用次数已用完。\n请使用「辉宝赐福」获取次数。")
                    return
                elif group_limit_on and group_count <= 0:
                    # 只开启群组限制，群组次数为0
                    yield event.plain_result("❌ 本群次数已用尽。\n请联系管理员增加群组次数。")
                    return
            else:
                # 私聊中：只检查用户限制
                if user_limit_on and user_count <= 0:
                    yield event.plain_result("❌ 您的使用次数已用完。\n请使用「辉宝赐福」获取次数。")
                    return
        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。");
                return
        images_to_process: List[bytes] = []
        initial_messages: List[str] = []
        display_cmd = cmd
        if is_bnn:
            MAX_IMAGES = 5
            original_count = len(img_bytes_list)
            if original_count > MAX_IMAGES:
                images_to_process = img_bytes_list[:MAX_IMAGES]
                initial_messages.append(f"🎨 检测到 {original_count} 张图片，已选取前 {MAX_IMAGES} 张…")
            else:
                images_to_process = img_bytes_list
            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
            initial_messages.append(f"🎨 检测到 {len(images_to_process)} 张图片，正在生成 [{display_cmd}]...")
        else:
            if not img_bytes_list:
                yield event.plain_result("请发送或引用一张图片。");
                return
            images_to_process = [img_bytes_list[0]]
            initial_messages.append(f"🎨 收到请求，正在生成 [{cmd}]...")

        slot_acquired = False
        try:
            if not await self._acquire_group_slot(group_id):
                if self.group_task_limit > 0:
                    yield event.plain_result(f"⚠️ 当前本群已有 {self.group_task_limit} 个生成任务正在处理，请稍后再试。")
                else:
                    yield event.plain_result("⚠️ 当前生成任务过多，请稍后再试。")
                return
            slot_acquired = True

            for msg in initial_messages:
                yield event.plain_result(msg)

            start_time = datetime.now()
            res = await self._call_api(images_to_process, user_prompt)
            elapsed = (datetime.now() - start_time).total_seconds()
            if isinstance(res, bytes):
                if not is_master:
                    if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                        await self._decrease_group_count(group_id)
                    elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                        await self._decrease_user_count(sender_id)
                caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {display_cmd}"]
                if is_master:
                    caption_parts.append("剩余次数: ∞")
                else:
                    if self.conf.get("enable_user_limit", True): caption_parts.append(
                        f"个人剩余: {self._get_user_count(sender_id)}")
                    if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                        f"本群剩余: {self._get_group_count(group_id)}")
                yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
            else:
                yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        finally:
            if slot_acquired:
                await self._release_group_slot(group_id)
        event.stop_event()

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image_request(self, event: AstrMessageEvent):
        prompt = event.message_str.strip()
        if not prompt:
            yield event.plain_result("请提供文生图的描述。用法: #文生图 <描述>")
            return

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)

        # --- 权限和次数检查 ---
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            
            # 修复次数限制逻辑（与图生图保持一致）
            if group_id:
                if user_limit_on and group_limit_on:
                    if user_count <= 0 and group_count <= 0:
                        yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。\n请使用「辉宝赐福」获取次数。")
                        return
                elif user_limit_on and user_count <= 0:
                    yield event.plain_result("❌ 您的使用次数已用完。\n请使用「辉宝赐福」获取次数。")
                    return
                elif group_limit_on and group_count <= 0:
                    yield event.plain_result("❌ 本群次数已用尽。\n请联系管理员增加群组次数。")
                    return
            else:
                if user_limit_on and user_count <= 0:
                    yield event.plain_result("❌ 您的使用次数已用完。\n请使用「辉宝赐福」获取次数。")
                    return

        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        slot_acquired = False
        try:
            if not await self._acquire_group_slot(group_id):
                if self.group_task_limit > 0:
                    yield event.plain_result(f"⚠️ 当前本群已有 {self.group_task_limit} 个生成任务正在处理，请稍后再试。")
                else:
                    yield event.plain_result("⚠️ 当前生成任务过多，请稍后再试。")
                return
            slot_acquired = True

            yield event.plain_result(f"🎨 收到文生图请求，正在生成 [{display_prompt}]...")

            start_time = datetime.now()
            # 调用通用API，但传入空的图片列表
            res = await self._call_api([], prompt)
            elapsed = (datetime.now() - start_time).total_seconds()

            if isinstance(res, bytes):
                if not is_master:
                    # 扣除次数
                    if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                        await self._decrease_group_count(group_id)
                    elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                        await self._decrease_user_count(sender_id)

                caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)"]
                if is_master:
                    caption_parts.append("剩余次数: ∞")
                else:
                    if self.conf.get("enable_user_limit", True): caption_parts.append(
                        f"个人剩余: {self._get_user_count(sender_id)}")
                    if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                        f"本群剩余: {self._get_group_count(group_id)}")
                yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
            else:
                yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        finally:
            if slot_acquired:
                await self._release_group_slot(group_id)
        event.stop_event()

    @filter.command("预设添加", aliases={"lm添加", "lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        raw = event.message_str.strip()
        if ":" not in raw:
            yield event.plain_result('格式错误, 正确示例:\n#预设添加 姿势表:为这幅图创建一个姿势表, 摆出各种姿势')
            return

        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break
        if not found: prompt_list.append(f"{key}:{new_value}")

        await self.conf.set("prompt_list", prompt_list)
        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存生图预设:\n{key}:{new_value}")

    @filter.command("生图帮助", aliases={"lm帮助", "lmh"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        raw_keyword = event.message_str.strip()

        # 兼容直接发送"生图帮助"而没有附加参数的情况
        keyword = raw_keyword
        for prefix_symbol in ("#", "/", "！", "!"):
            if keyword.startswith(prefix_symbol):
                keyword = keyword[len(prefix_symbol):].strip()
        if keyword in {"", "lm帮助", "lmh", "生图帮助"}:
            msg = "📸 【生图插件帮助】\n\n"
            msg += "🎨 图生图预设指令:\n"
            msg += "、".join(self.prompt_map.keys())
            msg += "\n\n✏️ 纯文本生图指令:\n#文生图 <你的描述>"
            msg += "\n\n💡 使用方法:\n发送图片 + 预设指令 或 @用户 + 预设指令 来进行图生图。"
            msg += "\n\n🎁 每日签到:\n发送「辉宝赐福」获取免费次数"
            yield event.plain_result(msg)
            return

        prompt = self.prompt_map.get(keyword)
        if not prompt:
            yield event.plain_result("❌ 未找到此预设指令")
            return
        yield event.plain_result(f"📋 预设 [{keyword}] 的内容:\n{prompt}")

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids

    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True);
            self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None,
                                                   functools.partial(json.dumps, self.user_counts, ensure_ascii=False,
                                                                     indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0: self.user_counts[user_id_str] = count - 1; await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists(): self.group_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.group_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.group_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True);
            self.group_counts = {}

    async def _save_group_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None,
                                                   functools.partial(json.dumps, self.group_counts, ensure_ascii=False,
                                                                     indent=4))
            await loop.run_in_executor(None, self.group_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        count = self._get_group_count(group_id_str)
        if count > 0: self.group_counts[group_id_str] = count - 1; await self._save_group_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists(): self.user_checkin_data = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_checkin_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_checkin_data = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户签到文件时发生错误: {e}", exc_info=True);
            self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_checkin_data,
                                                                           ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_checkin_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户签到文件时发生错误: {e}", exc_info=True)

    @filter.regex(r"^[#/!！]?辉宝赐福\s*$")
    async def on_checkin(self, event: AstrMessageEvent):
        """每日签到获取生图次数 - 支持直接发送"辉宝赐福"触发"""
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 本机器人未开启辉宝赐福功能。")
            return
        user_id = event.get_sender_id()
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.user_checkin_data.get(user_id) == today_str:
            yield event.plain_result(f"您今天已经领取过辉宝赐福。\n剩余次数: {self._get_user_count(user_id)}")
            return
        reward = 0
        if str(self.conf.get("enable_random_checkin", False)).lower() == 'true':
            max_reward = max(1, int(self.conf.get("checkin_random_reward_max", 5)))
            reward = random.randint(1, max_reward)
        else:
            reward = int(self.conf.get("checkin_fixed_reward", 3))
        current_count = self._get_user_count(user_id)
        new_count = current_count + reward
        self.user_counts[user_id] = new_count
        await self._save_user_counts()
        self.user_checkin_data[user_id] = today_str
        await self._save_user_checkin_data()
        yield event.plain_result(f"🎉 辉宝赐福成功！获得 {reward} 次，当前剩余: {new_count} 次。")

    @filter.command("生图增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        cmd_text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target_qq, count = None, 0
        if at_seg:
            target_qq = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", cmd_text)
            if match: count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", cmd_text)
            if match: target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0:
            yield event.plain_result(
                '格式错误:\n#生图增加用户次数 @用户 <次数>\n或 #生图增加用户次数 <QQ号> <次数>')
            return
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("生图增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if not match:
            yield event.plain_result('格式错误: #生图增加群组次数 <群号> <次数>')
            return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("生图查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                user_id_to_query = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match: user_id_to_query = match.group(1)
        user_count = self._get_user_count(user_id_to_query)
        reply_msg = f"用户 {user_id_to_query} 个人剩余次数为: {user_count}"
        if user_id_to_query == event.get_sender_id(): reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        if group_id := event.get_group_id(): reply_msg += f"\n本群共享剩余次数为: {self._get_group_count(group_id)}"
        yield event.plain_result(reply_msg)

    @filter.command("生图添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        new_keys = event.message_str.strip().split()
        if not new_keys: yield event.plain_result("格式错误，请提供要添加的Key。"); return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("生图key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: yield event.plain_result("📝 暂未配置任何 API Key。"); return
        key_list_str = "\n".join(f"{i + 1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("生图删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            await self.conf.set("api_keys", [])
            yield event.plain_result(f"✅ 已删除全部 {len(api_keys)} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            removed_key = api_keys.pop(int(param) - 1)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else:
            yield event.plain_result("格式错误，请使用 #生图删除key <序号|all>")

    async def _get_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    async def _acquire_group_slot(self, group_id: Optional[str]) -> bool:
        if not group_id or self.group_task_limit <= 0:
            return True
        async with self.queue_lock:
            current = self.group_task_counts.get(group_id, 0)
            if current >= self.group_task_limit:
                return False
            self.group_task_counts[group_id] = current + 1
            logger.debug(f"[FigurinePro] 群 {group_id} 任务占用 {self.group_task_counts[group_id]}/{self.group_task_limit}")
            return True

    async def _release_group_slot(self, group_id: Optional[str]):
        if not group_id or self.group_task_limit <= 0:
            return
        async with self.queue_lock:
            current = self.group_task_counts.get(group_id, 0)
            if current <= 1:
                self.group_task_counts.pop(group_id, None)
            else:
                self.group_task_counts[group_id] = current - 1
            logger.debug(f"[FigurinePro] 群 {group_id} 任务释放，当前 {self.group_task_counts.get(group_id, 0)}")


    async def _extract_image_from_markdown(self, text: str) -> bytes | None:
        if not text or not self.iwf:
            return None
        match = re.search(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", text)
        if match:
            url = match.group(1).strip()
            try:
                downloaded = await self.iwf._download_image(url)
                if downloaded:
                    return downloaded
            except Exception as e:
                logger.warning(f"下载 Markdown 图片失败: {e}", exc_info=True)
        return None

    async def _extract_image_bytes_from_response(self, data: Dict[str, Any]) -> bytes | None:
        """
        从 OpenAI / OpenRouter 风格的响应中提取图像数据。
        兼容多种可能的返回结构，包括：
            - images 直接列表
            - chat.completions 的 choices[].message.images
            - chat.completions 的 choices[].message.content 内嵌
            - data[].url / data[].b64_json
        """
        try:
            # 1. OpenAI Images API 风格 {"data": [{"url": "..."}]} 或 {"data": [{"b64_json": "..."}]}
            if isinstance(data.get("data"), list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        if url := item.get("url"):
                            if url.startswith("data:image/"):
                                return base64.b64decode(url.split(",", 1)[1])
                            if self.iwf:
                                downloaded = await self.iwf._download_image(url)
                                if downloaded:
                                    return downloaded
                        if b64 := item.get("b64_json"):
                            return base64.b64decode(b64)

            # 2. 旧格式 {"images": [{"url": "..."}]}
            if isinstance(data.get("images"), list):
                for image in data["images"]:
                    if not isinstance(image, dict):
                        continue
                    url = image.get("url")
                    if url:
                        if url.startswith("data:image/"):
                            return base64.b64decode(url.split(",", 1)[1])
                        if self.iwf:
                            downloaded = await self.iwf._download_image(url)
                            if downloaded:
                                return downloaded

            # 3. Chat Completions 风格
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})

                # 3.1 message.images 显式结构
                if isinstance(message.get("images"), list):
                    for image in message["images"]:
                        if not isinstance(image, dict):
                            continue
                        url = image.get("image_url", {}).get("url") or image.get("url")
                        if url:
                            if url.startswith("data:image/"):
                                return base64.b64decode(url.split(",", 1)[1])
                            if self.iwf:
                                downloaded = await self.iwf._download_image(url)
                                if downloaded:
                                    return downloaded

                content = message.get("content")

                # 3.2 content 为列表（OpenAI 新版多模态结构）
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue

                        item_type = item.get("type")
                        if item_type in {"output_image", "image_url", "image"}:
                            if isinstance(item.get("image_url"), dict):
                                url = item["image_url"].get("url")
                                if url:
                                    if url.startswith("data:image/"):
                                        return base64.b64decode(url.split(",", 1)[1])
                                    if self.iwf:
                                        downloaded = await self.iwf._download_image(url)
                                        if downloaded:
                                            return downloaded
                            if url := item.get("url"):
                                if url.startswith("data:image/"):
                                    return base64.b64decode(url.split(",", 1)[1])
                                if self.iwf:
                                    downloaded = await self.iwf._download_image(url)
                                    if downloaded:
                                        return downloaded
                            if b64 := item.get("b64_json"):
                                return base64.b64decode(b64)

                        # 兼容部分模型直接返回 base64 文本
                        if item_type in {"text", "output_text"} and isinstance(item.get("text"), str):
                            text_content = item["text"]
                            matches = re.findall(r"data:image/([^;]+);base64,([A-Za-z0-9+/=]+)", text_content)
                            if matches:
                                return base64.b64decode(matches[0][1])
                            markdown_img = await self._extract_image_from_markdown(text_content)
                            if markdown_img:
                                return markdown_img

                # 3.3 content 为字符串，尝试匹配其中的 base64
                if isinstance(content, str):
                    matches = re.findall(r"data:image/([^;]+);base64,([A-Za-z0-9+/=]+)", content)
                    if matches:
                        return base64.b64decode(matches[0][1])
                    markdown_img = await self._extract_image_from_markdown(content)
                    if markdown_img:
                        return markdown_img

            logger.warning(f"未能在响应中提取图像数据，原始响应(截断): {str(data)[:200]}")
            return None
        except Exception as e:
            logger.error(f"解析图像响应时出现错误: {e}", exc_info=True)
            return None

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        """调用 API 生成图像，优先使用选择的供应商，否则使用手动配置"""
        
        # 获取模型名称（必填）
        model_name = self.conf.get("model", "").strip()
        if not model_name:
            return "❌ 模型名称 (model) 未配置"
        
        # 确定 API URL 和 Key
        api_url: str = ""
        api_key: str = ""
        
        # 优先使用供应商配置
        if self.provider_id and self.provider:
            # 从供应商获取配置
            try:
                provider_config = self.provider.get_config() if hasattr(self.provider, 'get_config') else {}
                api_url = getattr(self.provider, 'api_base', '') or provider_config.get('api_base', '') or provider_config.get('base_url', '')
                api_key = getattr(self.provider, 'api_key', '') or provider_config.get('api_key', '') or provider_config.get('key', '')
                
                # 尝试从不同属性获取
                if not api_url:
                    for attr in ['base_url', 'api_url', 'endpoint']:
                        if hasattr(self.provider, attr):
                            api_url = getattr(self.provider, attr, '')
                            if api_url:
                                break
                
                if not api_key:
                    for attr in ['key', 'secret_key', 'token']:
                        if hasattr(self.provider, attr):
                            api_key = getattr(self.provider, attr, '')
                            if api_key:
                                break
                
                if api_url:
                    logger.debug(f"使用提供商 '{self.provider_id}' 的 API: {api_url[:50]}...")
            except Exception as e:
                logger.warning(f"从提供商获取配置失败: {e}，将尝试使用手动配置")
        
        # 如果供应商没有提供有效配置，使用手动配置
        if not api_url:
            api_url_raw = (self.conf.get("api_url") or "").strip()
            if not api_url_raw:
                return "❌ 未选择提供商，且未配置 API URL"
            api_url = api_url_raw
        
        if not api_key:
            api_key = await self._get_api_key()
            if not api_key:
                return "❌ 未选择提供商，且未配置 API Key"
        
        # 处理 API URL 格式
        if not re.search(r"/v\d+/(chat|images)/", api_url):
            api_url = api_url.rstrip("/") + "/v1/chat/completions"
            logger.debug(f"自动拼接完整 API 路径: {api_url}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/astrbot",
            "X-Title": "AstrBot NewImage Plugin",
        }

        message_content: List[Dict[str, Any]] = []
        if prompt:
            message_content.append({"type": "text", "text": prompt})

        if image_bytes_list:
            try:
                for idx, img_bytes in enumerate(image_bytes_list):
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )
            except Exception as e:
                logger.error(f"Base64 编码图片时出错: {e}", exc_info=True)
                return f"图片编码失败: {e}"

        if not message_content:
            return "缺少 prompt 或图片内容"

        if len(message_content) == 1 and message_content[0].get("type") == "text":
            user_content: Any = message_content[0]["text"]
        else:
            user_content = message_content

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }

        source_info = f"提供商:{self.provider_id}" if (self.provider_id and self.provider) else "手动配置"
        logger.info(f"[NewImage] 发送请求 [{source_info}]: Model={model_name}, HasImage={bool(image_bytes_list)}")

        try:
            if not self.iwf: return "ImageWorkflow 未初始化"
            async with self.iwf.session.post(
                api_url,
                json=payload,
                headers=headers,
                proxy=self.iwf.proxy,
                timeout=120,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"

                data = await resp.json()

                image_bytes = await self._extract_image_bytes_from_response(data)

                if image_bytes:
                    return image_bytes

                if "error" in data:
                    return data["error"].get("message", json.dumps(data["error"]))

                error_msg = f"API响应中未找到可用的图像数据: {str(data)[:500]}..."
                logger.error(error_msg)
                return error_msg
        except asyncio.TimeoutError:
            logger.error("API 请求超时");
            return "请求超时"
        except Exception as e:
            logger.error(f"调用 API 时发生未知错误: {e}", exc_info=True);
            return f"发生未知错误: {e}"

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
