"""
desktop_app.py
Interfaz Gráfica de Escritorio Nativa para Windows con CustomTkinter.
Incluye dos herramientas integradas:
1. Detector de Inteligencia Artificial con Red Neuronal y Mapa de Calor.
2. Humanizador de Texto Antidetección con autoverificación en tiempo real.
"""

import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from detector.engine import detector_engine
from detector.file_reader import extract_text_from_file
from detector.humanizer import humanize_text

# Configuración visual
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AIDetectorDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VeritasAI Pro - Detector & Humanizador de IA")
        self.geometry("1150x760")
        self.minsize(1000, 680)

        self.create_widgets()

    def create_widgets(self):
        # Header Superior
        header_frame = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        header_frame.pack(fill="x", padx=15, pady=(15, 8))

        title_label = ctk.CTkLabel(
            header_frame,
            text="🧠 VeritasAI Pro Ultra",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#60a5fa"
        )
        title_label.pack(side="left", padx=15, pady=10)

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Detector Multicapa & Humanizador Antidetección (ChatGPT • Gemini • Claude • Perplexity)",
            font=ctk.CTkFont(size=12),
            text_color="#9ca3af"
        )
        subtitle_label.pack(side="left", padx=5, pady=10)

        # Tabview principal
        self.tabview = ctk.CTkTabview(self, corner_radius=12)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.tab_detector = self.tabview.add("🔍 Detector de IA")
        self.tab_humanizer = self.tabview.add("✨ Humanizador de Texto")

        self.build_detector_tab()
        self.build_humanizer_tab()

    # =========================================================================
    # TAB 1: DETECTOR DE IA
    # =========================================================================
    def build_detector_tab(self):
        main_pane = ctk.CTkFrame(self.tab_detector, fg_color="transparent")
        main_pane.pack(fill="both", expand=True, padx=5, pady=5)

        # Panel Izquierdo (Entrada)
        left_frame = ctk.CTkFrame(main_pane, corner_radius=10)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        sample_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
        sample_bar.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(sample_bar, text="Texto a analizar:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        btn_sample_human = ctk.CTkButton(sample_bar, text="Ej. Humano", width=75, height=24, fg_color="#059669", hover_color="#047857", command=self.load_sample_human)
        btn_sample_human.pack(side="right", padx=2)

        btn_sample_gemini = ctk.CTkButton(sample_bar, text="Ej. Gemini", width=75, height=24, fg_color="#0284c7", hover_color="#0369a1", command=self.load_sample_gemini)
        btn_sample_gemini.pack(side="right", padx=2)

        btn_sample_chatgpt = ctk.CTkButton(sample_bar, text="Ej. ChatGPT", width=75, height=24, fg_color="#4f46e5", hover_color="#4338ca", command=self.load_sample_chatgpt)
        btn_sample_chatgpt.pack(side="right", padx=2)

        self.text_input = ctk.CTkTextbox(left_frame, font=ctk.CTkFont(family="Consolas", size=12), wrap="word", corner_radius=8)
        self.text_input.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_input.insert("1.0", "Pega aquí el texto a analizar o carga un archivo PDF, Word o TXT...")

        actions_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
        actions_bar.pack(fill="x", padx=10, pady=(5, 10))

        btn_file = ctk.CTkButton(actions_bar, text="📁 Cargar Archivo", width=110, height=34, fg_color="#374151", hover_color="#4b5563", command=self.upload_file)
        btn_file.pack(side="left", padx=(0, 6))

        btn_clear = ctk.CTkButton(actions_bar, text="🗑️ Limpiar", width=70, height=34, fg_color="#374151", hover_color="#4b5563", command=self.clear_all)
        btn_clear.pack(side="left")

        self.btn_analyze = ctk.CTkButton(actions_bar, text="🚀 ANALIZAR TEXTO", height=36, font=ctk.CTkFont(size=13, weight="bold"), fg_color="#4f46e5", hover_color="#4338ca", command=self.start_analysis_thread)
        self.btn_analyze.pack(side="right", fill="x", expand=True, padx=(8, 0))

        # Panel Derecho (Resultados)
        right_frame = ctk.CTkFrame(main_pane, corner_radius=10, width=400)
        right_frame.pack(side="right", fill="both", padx=(8, 0), pady=0)
        right_frame.pack_propagate(False)

        self.card_verdict = ctk.CTkFrame(right_frame, fg_color="#1f2937", corner_radius=10)
        self.card_verdict.pack(fill="x", padx=10, pady=(10, 6))

        self.lbl_verdict_badge = ctk.CTkLabel(self.card_verdict, text="ESPERANDO ANÁLISIS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#9ca3af")
        self.lbl_verdict_badge.pack(pady=(6, 2))

        self.lbl_ai_prob = ctk.CTkLabel(self.card_verdict, text="0%", font=ctk.CTkFont(size=32, weight="bold"), text_color="#e5e7eb")
        self.lbl_ai_prob.pack(pady=0)

        self.progress_ai = ctk.CTkProgressBar(self.card_verdict, height=8, corner_radius=4)
        self.progress_ai.pack(fill="x", padx=16, pady=(4, 6))
        self.progress_ai.set(0)

        self.lbl_verdict_desc = ctk.CTkLabel(self.card_verdict, text="Pega un texto y haz clic en Analizar Texto.", font=ctk.CTkFont(size=11), text_color="#d1d5db", wraplength=340)
        self.lbl_verdict_desc.pack(padx=10, pady=(0, 8))

        # Botón para enviar a humanizar si es IA
        self.btn_send_to_h = ctk.CTkButton(right_frame, text="✨ Humanizar este texto", height=28, fg_color="#d97706", hover_color="#b45309", font=ctk.CTkFont(size=11, weight="bold"), command=self.transfer_to_humanizer)
        self.btn_send_to_h.pack(fill="x", padx=10, pady=4)

        # Fila de métricas
        metrics_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=10, pady=2)

        box_burst = ctk.CTkFrame(metrics_frame, fg_color="#1f2937", corner_radius=6)
        box_burst.pack(side="left", fill="both", expand=True, padx=(0, 3))
        ctk.CTkLabel(box_burst, text="Burstiness", font=ctk.CTkFont(size=10, weight="bold"), text_color="#9ca3af").pack(pady=(2, 0))
        self.lbl_val_burst = ctk.CTkLabel(box_burst, text="--", font=ctk.CTkFont(size=12, weight="bold"), text_color="#60a5fa")
        self.lbl_val_burst.pack(pady=(0, 2))

        box_perp = ctk.CTkFrame(metrics_frame, fg_color="#1f2937", corner_radius=6)
        box_perp.pack(side="left", fill="both", expand=True, padx=(3, 3))
        ctk.CTkLabel(box_perp, text="Perplejidad", font=ctk.CTkFont(size=10, weight="bold"), text_color="#9ca3af").pack(pady=(2, 0))
        self.lbl_val_perp = ctk.CTkLabel(box_perp, text="--", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8")
        self.lbl_val_perp.pack(pady=(0, 2))

        box_cliche = ctk.CTkFrame(metrics_frame, fg_color="#1f2937", corner_radius=6)
        box_cliche.pack(side="left", fill="both", expand=True, padx=(3, 0))
        ctk.CTkLabel(box_cliche, text="Clichés IA", font=ctk.CTkFont(size=10, weight="bold"), text_color="#9ca3af").pack(pady=(2, 0))
        self.lbl_val_cliche = ctk.CTkLabel(box_cliche, text="--", font=ctk.CTkFont(size=12, weight="bold"), text_color="#fbbf24")
        self.lbl_val_cliche.pack(pady=(0, 2))

        self.lbl_model_signature = ctk.CTkLabel(right_frame, text="Firma: Sin analizar", font=ctk.CTkFont(size=10, weight="bold"), text_color="#c084fc")
        self.lbl_model_signature.pack(pady=2)

        ctk.CTkLabel(right_frame, text="Mapa de calor por oración:", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(4, 2))

        self.scroll_sentences = ctk.CTkScrollableFrame(right_frame, fg_color="#111827", corner_radius=8)
        self.scroll_sentences.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # =========================================================================
    # TAB 2: HUMANIZADOR DE TEXTO
    # =========================================================================
    def build_humanizer_tab(self):
        h_main = ctk.CTkFrame(self.tab_humanizer, fg_color="transparent")
        h_main.pack(fill="both", expand=True, padx=5, pady=5)

        # Panel Entrada Humanizador (Izquierda)
        h_left = ctk.CTkFrame(h_main, corner_radius=10)
        h_left.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        h_top_bar = ctk.CTkFrame(h_left, fg_color="transparent")
        h_top_bar.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(h_top_bar, text="Texto original de IA:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        btn_paste_from_d = ctk.CTkButton(h_top_bar, text="📋 Pegar del Detector", width=120, height=24, fg_color="#374151", hover_color="#4b5563", command=self.paste_detector_into_humanizer)
        btn_paste_from_d.pack(side="right")

        self.h_text_input = ctk.CTkTextbox(h_left, font=ctk.CTkFont(family="Consolas", size=12), wrap="word", corner_radius=8)
        self.h_text_input.pack(fill="both", expand=True, padx=10, pady=5)
        self.h_text_input.insert("1.0", "Pega aquí el texto generado por IA que deseas humanizar...")

        h_controls = ctk.CTkFrame(h_left, fg_color="transparent")
        h_controls.pack(fill="x", padx=10, pady=(4, 10))

        ctk.CTkLabel(h_controls, text="Modo:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 4))
        self.combo_mode = ctk.CTkComboBox(h_controls, values=["Stealth (Antidetección)", "Académico Natural", "Conversacional"], width=190, height=32)
        self.combo_mode.set("Stealth (Antidetección)")
        self.combo_mode.pack(side="left")

        self.btn_run_humanize = ctk.CTkButton(h_controls, text="✨ HUMANIZAR TEXTO", height=34, font=ctk.CTkFont(size=12, weight="bold"), fg_color="#d97706", hover_color="#b45309", command=self.start_humanize_thread)
        self.btn_run_humanize.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Panel Salida Humanizador (Derecha)
        h_right = ctk.CTkFrame(h_main, corner_radius=10, width=440)
        h_right.pack(side="right", fill="both", padx=(8, 0), pady=0)
        h_right.pack_propagate(False)

        # Tarjeta de comparación Antes/Después
        self.card_h_stats = ctk.CTkFrame(h_right, fg_color="#1f2937", corner_radius=10)
        self.card_h_stats.pack(fill="x", padx=10, pady=(10, 6))

        self.lbl_h_status = ctk.CTkLabel(self.card_h_stats, text="ESPERANDO TEXTO", font=ctk.CTkFont(size=11, weight="bold"), text_color="#9ca3af")
        self.lbl_h_status.pack(pady=(6, 2))

        self.lbl_h_reduction = ctk.CTkLabel(self.card_h_stats, text="--", font=ctk.CTkFont(size=22, weight="bold"), text_color="#10b981")
        self.lbl_h_reduction.pack(pady=0)

        self.lbl_h_details = ctk.CTkLabel(self.card_h_stats, text="Presiona Humanizar Texto para procesar.", font=ctk.CTkFont(size=11), text_color="#d1d5db")
        self.lbl_h_details.pack(padx=10, pady=(2, 8))

        ctk.CTkLabel(h_right, text="Texto Humanizado Resultante:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=12, pady=(4, 2))

        self.h_text_output = ctk.CTkTextbox(h_right, font=ctk.CTkFont(family="Consolas", size=12), wrap="word", corner_radius=8)
        self.h_text_output.pack(fill="both", expand=True, padx=10, pady=5)

        h_out_actions = ctk.CTkFrame(h_right, fg_color="transparent")
        h_out_actions.pack(fill="x", padx=10, pady=(4, 10))

        btn_copy_h = ctk.CTkButton(h_out_actions, text="📋 Copiar", width=80, height=32, fg_color="#374151", hover_color="#4b5563", command=self.copy_humanized)
        btn_copy_h.pack(side="left", padx=(0, 4))

        btn_save_h = ctk.CTkButton(h_out_actions, text="💾 Guardar TXT", width=95, height=32, fg_color="#374151", hover_color="#4b5563", command=self.save_humanized_file)
        btn_save_h.pack(side="left")

        btn_test_in_d = ctk.CTkButton(h_out_actions, text="⚡ Probar en Detector", height=32, fg_color="#4f46e5", hover_color="#4338ca", font=ctk.CTkFont(size=11, weight="bold"), command=self.send_humanized_to_detector)
        btn_test_in_d.pack(side="right", fill="x", expand=True, padx=(8, 0))

    # =========================================================================
    # LÓGICA DEL DETECTOR
    # =========================================================================
    def clear_all(self):
        self.text_input.delete("1.0", "end")
        self.lbl_verdict_badge.configure(text="ESPERANDO ANÁLISIS", text_color="#9ca3af")
        self.lbl_ai_prob.configure(text="0%", text_color="#e5e7eb")
        self.progress_ai.set(0)
        self.lbl_verdict_desc.configure(text="Pega un texto y haz clic en Analizar Texto.")
        self.lbl_val_burst.configure(text="--")
        self.lbl_val_perp.configure(text="--")
        self.lbl_val_cliche.configure(text="--")
        self.lbl_model_signature.configure(text="Firma: Sin analizar")
        for widget in self.scroll_sentences.winfo_children():
            widget.destroy()

    def load_sample_chatgpt(self):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", 
            "En un mundo cada vez más digitalizado, la inteligencia artificial desempeña un papel fundamental "
            "en la transformación de los procesos productivos. Cabe destacar que, si bien ofrece ventajas "
            "significativas, es crucial sopesar los riesgos éticos asociados. En conclusión, debemos forjar un futuro responsable."
        )

    def load_sample_gemini(self):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", 
            "Aquí tienes un desglose de los puntos clave sobre la computación cuántica:\n"
            "- Permite procesar datos mediante cúbits con superposición.\n"
            "- En términos generales, acelera la simulación molecular.\n"
            "- En resumen, revolucionará la criptografía moderna."
        )

    def load_sample_human(self):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", 
            "Ayer por la tarde intenté preparar pan casero y me quedó como un ladrillo. "
            "Creo que la levadura estaba vencida o el horno calentó de más porque se quemó la base "
            "y por dentro parecía engrudo. Al final le eché la culpa al clima y terminé pidiendo pizza por teléfono."
        )

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo para analizar",
            filetypes=[("Documentos", "*.txt;*.pdf;*.docx;*.doc"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            filename = file_path.split("/")[-1].split("\\")[-1]
            success, text_or_err = extract_text_from_file(filename, content)
            if success:
                self.text_input.delete("1.0", "end")
                self.text_input.insert("1.0", text_or_err)
                messagebox.showinfo("Archivo Cargado", f"Se extrajo el texto de '{filename}' correctamente.")
            else:
                messagebox.showerror("Error", text_or_err)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el archivo: {e}")

    def start_analysis_thread(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Atención", "Por favor ingresa texto para analizar.")
            return

        self.btn_analyze.configure(state="disabled", text="⏳ Analizando...")
        thread = threading.Thread(target=self._run_analysis, args=(text,), daemon=True)
        thread.start()

    def _run_analysis(self, text):
        try:
            res = detector_engine.detect(text)
            self.after(0, self._render_results, res)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Fallo al analizar: {e}"))
            self.after(0, lambda: self.btn_analyze.configure(state="normal", text="🚀 ANALIZAR TEXTO"))

    def _render_results(self, res):
        self.btn_analyze.configure(state="normal", text="🚀 ANALIZAR TEXTO")

        if "error" in res:
            messagebox.showwarning("Aviso", res["error"])
            return

        ai_prob = res["ai_probability"]
        self.lbl_ai_prob.configure(text=f"{ai_prob}%")
        self.progress_ai.set(ai_prob / 100.0)

        if ai_prob >= 75.0:
            badge_color = "#f43f5e"
            prog_color = "#f43f5e"
        elif ai_prob >= 45.0:
            badge_color = "#f59e0b"
            prog_color = "#f59e0b"
        else:
            badge_color = "#10b981"
            prog_color = "#10b981"

        self.lbl_verdict_badge.configure(text=res["verdict"]["title"].upper(), text_color=badge_color)
        self.progress_ai.configure(progress_color=prog_color)
        self.lbl_verdict_desc.configure(text=res["verdict"]["description"])

        self.lbl_val_burst.configure(text=f"{res['metrics']['burstiness']['value']}")
        self.lbl_val_perp.configure(text=f"{res['metrics']['perplexity']['value']}")
        self.lbl_val_cliche.configure(text=f"{res['metrics']['ai_phrases_detected']['count']}")
        self.lbl_model_signature.configure(text=f"Firma: {res['metrics']['dominant_model']}")

        for widget in self.scroll_sentences.winfo_children():
            widget.destroy()

        for s in res["sentence_heatmap"]:
            color = "#ef4444" if s["label"] == "danger" else ("#f59e0b" if s["label"] == "warning" else "#10b981")
            
            f_sent = ctk.CTkFrame(self.scroll_sentences, fg_color="#1f2937", corner_radius=6)
            f_sent.pack(fill="x", pady=2, padx=2)

            lbl_tag = ctk.CTkLabel(f_sent, text=f"[{s['ai_probability']}% IA] {s['tag']}", font=ctk.CTkFont(size=10, weight="bold"), text_color=color)
            lbl_tag.pack(anchor="w", padx=6, pady=(3, 0))

            lbl_txt = ctk.CTkLabel(f_sent, text=s["text"], font=ctk.CTkFont(size=11), text_color="#e5e7eb", wraplength=340, justify="left")
            lbl_txt.pack(anchor="w", padx=6, pady=(0, 3))

    def transfer_to_humanizer(self):
        text = self.text_input.get("1.0", "end").strip()
        if text:
            self.h_text_input.delete("1.0", "end")
            self.h_text_input.insert("1.0", text)
            self.tabview.set("✨ Humanizador de Texto")

    def paste_detector_into_humanizer(self):
        self.transfer_to_humanizer()

    # =========================================================================
    # LÓGICA DEL HUMANIZADOR
    # =========================================================================
    def start_humanize_thread(self):
        text = self.h_text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Atención", "Por favor ingresa texto para humanizar.")
            return

        mode_map = {
            "Stealth (Antidetección)": "stealth",
            "Académico Natural": "academic",
            "Conversacional": "conversational"
        }
        mode = mode_map.get(self.combo_mode.get(), "stealth")

        self.btn_run_humanize.configure(state="disabled", text="⏳ Humanizando...")
        thread = threading.Thread(target=self._run_humanization, args=(text, mode), daemon=True)
        thread.start()

    def _run_humanization(self, text, mode):
        try:
            res = humanize_text(text, mode=mode)
            self.after(0, self._render_humanized_results, res)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Fallo al humanizar: {e}"))
            self.after(0, lambda: self.btn_run_humanize.configure(state="normal", text="✨ HUMANIZAR TEXTO"))

    def _render_humanized_results(self, res):
        self.btn_run_humanize.configure(state="normal", text="✨ HUMANIZAR TEXTO")
        if "error" in res:
            messagebox.showwarning("Aviso", res["error"])
            return

        self.h_text_output.delete("1.0", "end")
        self.h_text_output.insert("1.0", res["humanized_text"])

        ba = res["before_after"]
        self.lbl_h_status.configure(text="VERIFICACIÓN COMPLETADA", text_color="#10b981")
        self.lbl_h_reduction.configure(text=f"{ba['original_ai_prob']}% IA ➔ {ba['humanized_ai_prob']}% IA ({ba['ai_reduction']})")
        self.lbl_h_details.configure(text=f"Método: {res['method_used']} • Clichés eliminados: {res['cliches_removed']}")

    def copy_humanized(self):
        txt = self.h_text_output.get("1.0", "end").strip()
        if txt:
            self.clipboard_clear()
            self.clipboard_append(txt)
            messagebox.showinfo("Copiado", "Texto humanizado copiado al portapapeles.")

    def save_humanized_file(self):
        txt = self.h_text_output.get("1.0", "end").strip()
        if not txt:
            messagebox.showwarning("Atención", "No hay texto humanizado para guardar.")
            return
        fpath = filedialog.asksaveasfilename(
            title="Guardar texto humanizado",
            defaultextension=".txt",
            filetypes=[("Archivo de texto (*.txt)", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if fpath:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Guardado", f"Archivo guardado exitosamente.")

    def send_humanized_to_detector(self):
        txt = self.h_text_output.get("1.0", "end").strip()
        if txt:
            self.text_input.delete("1.0", "end")
            self.text_input.insert("1.0", txt)
            self.tabview.set("🔍 Detector de IA")
            self.start_analysis_thread()

if __name__ == "__main__":
    app = AIDetectorDesktopApp()
    app.mainloop()
