"""Desktop interface for the PlantVillage leaf-disease classifier."""

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from leaf_model import LABELS, display_label, load_model, predict_image


COLORS = {
    "background": "#F4F7F2",
    "surface": "#FFFFFF",
    "surface_soft": "#F8FAF7",
    "border": "#E4EAE1",
    "ink": "#17251D",
    "muted": "#64736B",
    "green": "#286B49",
    "green_dark": "#194D36",
    "green_soft": "#E9F3EC",
    "lime": "#C7E38A",
    "amber": "#A35C2F",
    "amber_soft": "#FFF1E8",
    "track": "#EAF0EA",
}

class LeafHealthApp:
    def __init__(self, root):
        self.root = root
        self.selected_image = None
        self.preview_photo = None
        self.busy = False
        self.model_files_ready = all(
            (ROOT_DIR / name).is_file() for name in ("model1.json", "model1.h5")
        )
        self._layout_after_id = None
        self._is_stacked = None

        root.title("Verdant — Plant Health")
        root.geometry("1120x790")
        root.minsize(640, 620)
        root.configure(bg=COLORS["background"])

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Leaf.Horizontal.TProgressbar",
            troughcolor=COLORS["track"],
            background=COLORS["green"],
            bordercolor=COLORS["track"],
            lightcolor=COLORS["green"],
            darkcolor=COLORS["green"],
            thickness=7,
        )
        style.configure(
            "Leaf.Vertical.TScrollbar",
            background=COLORS["border"],
            troughcolor=COLORS["background"],
            bordercolor=COLORS["background"],
            arrowcolor=COLORS["muted"],
        )

        self._build_interface()
        root.bind("<Return>", self._on_return)
        root.bind("<Control-o>", self._on_open_shortcut)
        root.bind("<Configure>", self._on_root_configure)
        root.bind("<MouseWheel>", self._on_mousewheel)
        self._apply_responsive_layout()

    def _label(self, parent, text, *, size=10, color=None, weight="normal", **kwargs):
        return tk.Label(
            parent,
            text=text,
            bg=kwargs.pop("bg", COLORS["surface"]),
            fg=color or COLORS["ink"],
            font=("Segoe UI", size, weight),
            **kwargs,
        )

    def _card(self, parent):
        return tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )

    def _build_interface(self):
        viewport = tk.Frame(self.root, bg=COLORS["background"])
        viewport.pack(fill="both", expand=True)
        self.page_canvas = tk.Canvas(
            viewport,
            bg=COLORS["background"],
            highlightthickness=0,
            bd=0,
        )
        scrollbar = ttk.Scrollbar(
            viewport,
            orient="vertical",
            command=self.page_canvas.yview,
            style="Leaf.Vertical.TScrollbar",
        )
        self.page_canvas.configure(yscrollcommand=scrollbar.set)
        self.page_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.shell = tk.Frame(self.page_canvas, bg=COLORS["background"])
        self.page_window = self.page_canvas.create_window(
            (0, 0), window=self.shell, anchor="nw",
        )
        self.shell.bind("<Configure>", self._update_scroll_region)
        self.page_canvas.bind("<Configure>", self._resize_page)

        page = tk.Frame(self.shell, bg=COLORS["background"])
        page.pack(fill="both", expand=True, padx=34, pady=(22, 18))

        header = tk.Frame(page, bg=COLORS["background"])
        header.pack(fill="x", pady=(0, 29))

        mark = tk.Canvas(
            header, width=38, height=38, bg=COLORS["green_dark"],
            highlightthickness=0, bd=0,
        )
        mark.pack(side="left")
        mark.create_polygon(
            10, 25, 12, 14, 23, 8, 29, 10, 27, 21, 17, 28,
            fill=COLORS["lime"], smooth=True, splinesteps=24,
        )
        mark.create_line(12, 27, 26, 12, fill="#F4F8E8", width=1.5)

        brand = tk.Frame(header, bg=COLORS["background"])
        brand.pack(side="left", padx=(11, 0))
        self._label(brand, "VERDANT", size=12, weight="bold").pack(anchor="w")
        self._label(
            brand, "PLANT HEALTH STUDIO", size=8, color=COLORS["muted"],
            weight="bold",
        ).pack(anchor="w", pady=(2, 0))

        self.model_status = tk.Label(
            header,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
            padx=13,
            pady=8,
        )
        self.model_status.pack(side="right")
        self._set_idle_status()

        self.hero_title = self._label(
            page,
            "A clearer picture of plant health.",
            size=27,
            weight="bold",
        )
        self.hero_title.pack(anchor="w")
        self._label(
            page,
            "Add a leaf photo for an instant AI-assisted screening across five crops.",
            size=11,
            color=COLORS["muted"],
        ).pack(anchor="w", pady=(7, 22))

        workspace = tk.Frame(page, bg=COLORS["background"])
        workspace.pack(fill="both", expand=True)
        self.workspace = workspace
        workspace.grid_columnconfigure(0, weight=1, uniform="workspace")
        workspace.grid_columnconfigure(1, weight=1, uniform="workspace")
        workspace.grid_rowconfigure(0, weight=1)

        self.upload_card = self._build_upload_card(workspace)
        self.result_card = self._build_result_card(workspace)

        footer = tk.Frame(page, bg=COLORS["background"])
        footer.pack(fill="x", pady=(17, 0))
        self.footer = footer
        self.footer_crops = self._label(
            footer,
            "APPLE  ·  CORN  ·  GRAPE  ·  POTATO  ·  TOMATO",
            size=9,
            color=COLORS["muted"],
            weight="bold",
        )
        self.footer_note = self._label(
            footer,
            "Screening aid only — confirm concerns with a local expert.",
            size=9,
            color=COLORS["muted"],
        )
        self.footer_crops.grid(row=0, column=0, sticky="w")
        self.footer_note.grid(row=0, column=1, sticky="e")
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=1)

    def _build_upload_card(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        content = tk.Frame(card, bg=COLORS["surface"])
        content.pack(fill="both", expand=True, padx=20, pady=20)

        self._label(
            content, "01  /  IMAGE INPUT", size=9,
            color=COLORS["green"], weight="bold",
        ).pack(anchor="w")
        self._label(
            content, "Leaf sample", size=17, weight="bold",
        ).pack(anchor="w", pady=(9, 3))
        self._label(
            content, "Choose a clear, well-lit photo of a single leaf.",
            size=10, color=COLORS["muted"],
        ).pack(anchor="w")

        self.preview_canvas = tk.Canvas(
            content,
            height=270,
            bg=COLORS["surface_soft"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            bd=0,
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=(18, 14))
        self.preview_canvas.bind("<Configure>", self._render_preview)
        self._render_preview()

        self.image_name = self._label(
            content, "No image selected", size=10, color=COLORS["muted"],
            wraplength=420, justify="left",
        )
        self.image_name.pack(anchor="w", fill="x", pady=(0, 14))

        self.choose_button = tk.Button(
            content,
            text="Choose & analyze photo   →",
            command=self.choose_image,
            bg=COLORS["green"],
            activebackground=COLORS["green_dark"],
            fg="#FFFFFF",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=2,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["green"],
            cursor="hand2",
            padx=18,
            pady=12,
            takefocus=True,
        )
        self.choose_button.pack(fill="x")
        self._bind_button_hover(self.choose_button, COLORS["green"], COLORS["green_dark"])

        self._label(
            content,
            "JPG, JPEG, or PNG  ·  Enter or Ctrl+O to choose",
            size=9,
            color=COLORS["muted"],
        ).pack(anchor="center", pady=(10, 0))
        return card

    def _build_result_card(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        content = tk.Frame(card, bg=COLORS["surface"])
        content.pack(fill="both", expand=True, padx=20, pady=20)

        heading = tk.Frame(content, bg=COLORS["surface"])
        heading.pack(fill="x")
        self._label(
            heading, "02  /  ANALYSIS", size=9,
            color=COLORS["green"], weight="bold",
        ).pack(anchor="w")
        self._label(
            heading, "Diagnosis", size=17, weight="bold",
        ).pack(anchor="w", pady=(9, 3))
        self._label(
            heading, "Your result and the model's leading alternatives.",
            size=10, color=COLORS["muted"],
        ).pack(anchor="w")

        self.result_placeholder = tk.Frame(
            content, bg=COLORS["surface"], height=390,
        )
        self.result_placeholder.pack(fill="both", expand=True, pady=(14, 0))
        self.result_placeholder.pack_propagate(False)
        self._draw_empty_state(self.result_placeholder)

        self.result_content = tk.Frame(content, bg=COLORS["surface"])
        self.result_content.pack_forget()

        result_top = tk.Frame(self.result_content, bg=COLORS["surface"])
        result_top.pack(fill="x", pady=(16, 0))
        self.result_badge = tk.Label(
            result_top,
            text="",
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=5,
            anchor="w",
        )
        self.result_badge.pack(side="left")
        self.score_heading = self._label(
            result_top, "", size=9, color=COLORS["muted"], weight="bold",
        )
        self.score_heading.pack(side="right")

        self.crop_value = self._label(
            self.result_content, "", size=12, color=COLORS["green"],
            weight="bold",
        )
        self.crop_value.pack(anchor="w", pady=(19, 3))
        self.condition_value = self._label(
            self.result_content, "", size=21, weight="bold",
            wraplength=420, justify="left", anchor="w",
        )
        self.condition_value.pack(anchor="w")

        self.score_value = self._label(
            self.result_content, "", size=10, color=COLORS["muted"],
        )
        self.score_value.pack(anchor="w", pady=(12, 7))
        self.score_bar = ttk.Progressbar(
            self.result_content,
            style="Leaf.Horizontal.TProgressbar",
            maximum=100,
        )
        self.score_bar.pack(fill="x")

        divider = tk.Frame(self.result_content, bg=COLORS["border"], height=1)
        divider.pack(fill="x", pady=(20, 15))
        self._label(
            self.result_content, "OTHER LIKELY MATCHES",
            size=9, color=COLORS["muted"], weight="bold",
        ).pack(anchor="w", pady=(0, 9))

        self.alternative_rows = []
        for _ in range(2):
            row = tk.Frame(self.result_content, bg=COLORS["surface"])
            row.pack(fill="x", pady=5)
            name = self._label(
                row, "", size=10, wraplength=250, justify="left", anchor="w",
            )
            name.pack(side="left", anchor="w", fill="x", expand=True)
            score = self._label(row, "", size=9, color=COLORS["muted"])
            score.pack(side="right")
            bar = ttk.Progressbar(
                self.result_content,
                style="Leaf.Horizontal.TProgressbar",
                maximum=100,
            )
            bar.pack(fill="x", pady=(2, 4))
            self.alternative_rows.append((name, score, bar))

        self.result_note = self._label(
            self.result_content,
            "Model scores are an estimate, not a confirmed diagnosis.",
            size=9,
            color=COLORS["muted"],
            wraplength=420,
            justify="left",
        )
        self.result_note.pack(anchor="w", pady=(15, 0))
        return card

    def _draw_empty_state(self, parent):
        illustration = tk.Canvas(
            parent, width=110, height=110, bg=COLORS["surface"],
            highlightthickness=0,
        )
        illustration.pack(pady=(49, 16))
        illustration.create_oval(
            13, 13, 97, 97, fill=COLORS["green_soft"], outline="",
        )
        illustration.create_polygon(
            36, 76, 35, 54, 46, 34, 69, 27, 76, 32, 73, 55, 55, 73,
            fill="#A9C88E", smooth=True, splinesteps=24,
        )
        illustration.create_line(36, 78, 70, 38, fill=COLORS["green"], width=2)
        illustration.create_line(48, 63, 47, 49, fill=COLORS["green"], width=1)
        illustration.create_line(57, 54, 69, 52, fill=COLORS["green"], width=1)

        self.placeholder_title = self._label(
            parent, "Ready when you are", size=12, weight="bold",
        )
        self.placeholder_title.pack()
        self.placeholder_detail = self._label(
            parent,
            "Select a photo to see the analysis here.",
            size=10,
            color=COLORS["muted"],
            wraplength=400,
            justify="center",
        )
        self.placeholder_detail.pack(pady=(6, 0))

    def _render_preview(self, _event=None):
        canvas = self.preview_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 180)
        height = max(canvas.winfo_height(), 180)
        if self.selected_image is None:
            center_x, center_y = width / 2, height / 2
            canvas.create_oval(
                center_x - 37, center_y - 37, center_x + 37, center_y + 37,
                fill=COLORS["green_soft"], outline="",
            )
            canvas.create_polygon(
                center_x - 17, center_y + 19,
                center_x - 16, center_y - 2,
                center_x - 3, center_y - 20,
                center_x + 17, center_y - 25,
                center_x + 23, center_y - 20,
                center_x + 18, center_y + 1,
                center_x + 1, center_y + 17,
                fill="#A9C88E", smooth=True, splinesteps=24,
            )
            canvas.create_line(
                center_x - 17, center_y + 22,
                center_x + 16, center_y - 18,
                fill=COLORS["green"], width=2,
            )
            canvas.create_text(
                center_x, center_y + 62, text="Your leaf photo",
                fill=COLORS["muted"], font=("Segoe UI", 9),
            )
            return

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image_copy = self.selected_image.copy()
        image_copy.thumbnail((width - 18, height - 18), resampling)
        self.preview_photo = ImageTk.PhotoImage(image_copy)
        canvas.create_image(width / 2, height / 2, image=self.preview_photo)

    def _bind_button_hover(self, button, normal, hover):
        button.bind("<Enter>", lambda _event: button.configure(bg=hover))
        button.bind("<Leave>", lambda _event: button.configure(bg=normal))

    def choose_image(self):
        if self.busy:
            return
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a leaf photo",
            filetypes=[
                ("Leaf images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            with Image.open(file_path) as source:
                self.selected_image = source.convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Could not open image",
                "Choose a valid JPG or PNG image.\n\n{}".format(error),
                parent=self.root,
            )
            return

        self._render_preview()
        self.image_name.configure(text=Path(file_path).name, fg=COLORS["ink"])
        self._set_placeholder(
            "Analyzing this leaf…",
            "The CNN is comparing the image with {} plant conditions.".format(
                len(LABELS)
            ),
        )
        self._set_busy(True)
        self.root.update_idletasks()
        threading.Thread(
            target=self._predict_in_background,
            args=(self.selected_image.copy(),),
            daemon=True,
        ).start()

    def _predict_in_background(self, source_image):
        try:
            model = load_model()
            ranked = predict_image(source_image, model=model)
            self.root.after(0, self._show_prediction, ranked)
        except Exception as error:
            self.root.after(0, self._show_error, error)

    def _set_busy(self, busy):
        self.busy = busy
        if busy:
            self.choose_button.configure(
                text="Analyzing leaf…", state="disabled", bg=COLORS["green_dark"],
            )
            self.model_status.configure(
                text="●  ANALYZING IMAGE",
                bg=COLORS["amber_soft"],
                fg=COLORS["amber"],
            )
        else:
            self.choose_button.configure(
                text="Choose another photo   →" if self.selected_image else
                "Choose & analyze photo   →",
                state="normal",
                bg=COLORS["green"],
            )
            self._set_idle_status()

    def _set_idle_status(self):
        if self.model_files_ready:
            self.model_status.configure(
                text="●  MODEL FILES READY",
                bg=COLORS["green_soft"],
                fg=COLORS["green"],
            )
        else:
            self.model_status.configure(
                text="●  MODEL FILES NOT FOUND",
                bg=COLORS["amber_soft"],
                fg=COLORS["amber"],
            )

    def _set_placeholder(self, title, detail, *, error=False):
        self.result_content.pack_forget()
        self.result_placeholder.pack(fill="both", expand=True, pady=(14, 0))
        self.placeholder_title.configure(
            text=title,
            fg=COLORS["amber"] if error else COLORS["ink"],
        )
        self.placeholder_detail.configure(
            text=detail,
            fg=COLORS["amber"] if error else COLORS["muted"],
        )

    def _show_prediction(self, ranked):
        self._set_busy(False)
        predicted_label, confidence = ranked[0]
        crop, condition = predicted_label.split("___", 1)
        healthy = condition.strip().lower() == "healthy"

        self.result_placeholder.pack_forget()
        self.result_content.pack(fill="both", expand=True)
        self.result_badge.configure(
            text="HEALTHY SIGNAL" if healthy else "POTENTIAL DISEASE",
            bg=COLORS["green_soft"] if healthy else COLORS["amber_soft"],
            fg=COLORS["green"] if healthy else COLORS["amber"],
        )
        self.score_heading.configure(text="TOP MODEL SCORE")
        self.crop_value.configure(text=display_label(crop))
        self.condition_value.configure(text=display_label(condition))
        self.score_value.configure(
            text="Model score  ·  {:.1f}%".format(confidence * 100)
        )
        self.score_bar.configure(value=max(0, min(confidence * 100, 100)))

        for (name, score_label, bar), (label, score) in zip(
            self.alternative_rows, ranked[1:]
        ):
            name.configure(text=display_label(label.split("___", 1)[1]))
            score_label.configure(text="{:.1f}%".format(score * 100))
            bar.configure(value=max(0, min(score * 100, 100)))

        self.result_note.configure(
            text="Scores are model estimates, not calibrated probabilities. "
            "Confirm suspected disease with a local expert."
        )
        self._apply_responsive_layout()

    def _show_error(self, error):
        self._set_busy(False)
        self._set_placeholder(
            "Analysis couldn't finish",
            "Try again after checking the model files and Python dependencies.",
            error=True,
        )
        self.model_status.configure(
            text="●  ANALYSIS NEEDS ATTENTION",
            bg=COLORS["amber_soft"],
            fg=COLORS["amber"],
        )
        messagebox.showerror(
            "Analysis could not be completed",
            "{}\n\nCheck that the model files and Python dependencies are "
            "available, then try again.".format(error),
            parent=self.root,
        )

    def _resize_page(self, event):
        self.page_canvas.itemconfigure(self.page_window, width=event.width)
        self._schedule_responsive_layout()

    def _update_scroll_region(self, _event=None):
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))

    def _on_root_configure(self, event):
        if event.widget is self.root:
            self._schedule_responsive_layout()

    def _schedule_responsive_layout(self):
        if self._layout_after_id is not None:
            self.root.after_cancel(self._layout_after_id)
        self._layout_after_id = self.root.after_idle(self._apply_responsive_layout)

    def _apply_responsive_layout(self):
        self._layout_after_id = None
        if not hasattr(self, "workspace"):
            return

        viewport_width = self.page_canvas.winfo_width()
        stacked = viewport_width < 920
        if stacked != self._is_stacked:
            self._is_stacked = stacked
            if stacked:
                self.workspace.grid_columnconfigure(0, weight=1, uniform="workspace")
                self.workspace.grid_columnconfigure(1, weight=0, uniform="")
                self.workspace.grid_rowconfigure(0, weight=0)
                self.workspace.grid_rowconfigure(1, weight=0)
                self.upload_card.grid_configure(
                    row=0, column=0, padx=0, pady=(0, 14),
                )
                self.result_card.grid_configure(
                    row=1, column=0, padx=0, pady=0,
                )
                self.page_canvas.itemconfigure(
                    self.page_window, width=viewport_width,
                )
                self.footer_crops.grid_configure(row=0, column=0, sticky="w")
                self.footer_note.grid_configure(row=1, column=0, sticky="w")
                self.footer.grid_columnconfigure(1, weight=0)
                self.footer.grid_columnconfigure(0, weight=1)
                self.hero_title.configure(font=("Segoe UI", 24, "bold"))
            else:
                self.workspace.grid_columnconfigure(0, weight=1, uniform="workspace")
                self.workspace.grid_columnconfigure(1, weight=1, uniform="workspace")
                self.workspace.grid_rowconfigure(0, weight=1)
                self.workspace.grid_rowconfigure(1, weight=0)
                self.upload_card.grid_configure(
                    row=0, column=0, padx=(0, 10), pady=0,
                )
                self.result_card.grid_configure(
                    row=0, column=1, padx=(10, 0), pady=0,
                )
                self.footer_crops.grid_configure(row=0, column=0, sticky="w")
                self.footer_note.grid_configure(row=0, column=1, sticky="e")
                self.footer.grid_columnconfigure(0, weight=1)
                self.footer.grid_columnconfigure(1, weight=1)
                self.hero_title.configure(font=("Segoe UI", 27, "bold"))

        available_width = max(240, viewport_width - (48 if stacked else 78))
        card_text_width = (
            available_width - 48
            if stacked
            else max(200, available_width // 2 - 48)
        )
        self.condition_value.configure(wraplength=card_text_width)
        self.result_note.configure(wraplength=card_text_width)
        self.placeholder_detail.configure(
            wraplength=max(200, card_text_width - 20)
        )
        self.image_name.configure(wraplength=card_text_width)
        for name, _score, _bar in self.alternative_rows:
            name.configure(wraplength=max(120, card_text_width - 55))

    def _on_mousewheel(self, event):
        self.page_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_return(self, _event):
        if self.root.focus_get() is self.choose_button:
            return
        self.choose_image()

    def _on_open_shortcut(self, _event):
        self.choose_image()
        return "break"


if __name__ == "__main__":
    window = tk.Tk()
    LeafHealthApp(window)
    window.mainloop()
