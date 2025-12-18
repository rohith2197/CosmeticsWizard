import customtkinter as ctk
from ModelTesting import predict_and_explain

# Custom Tkinter Theme and Colors
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BRIGHT_PURPLE = "#150070"
LIGHT_PURPLE = "#0c0040"
BG_WHITE = "#F5F5F7"
TEXT_BLACK = "#000000"

# Font Variables for different text
FONT_HEADER = ("Inter", 48, "bold")
FONT_SUBHEADER = ("Inter", 28, "bold")
FONT_LABEL = ("Inter", 24)
FONT_INFO_TITLE = ("Inter", 16, "bold")
FONT_INFO_CONTENT = ("Inter", 24)
FONT_BUTTON = ("Inter", 16, "bold")
FONT_RATING = ("Inter", 48, "bold")
FONT_RESULT_TITLE = ("Inter", 24, "bold")
FONT_RESULT_CONTENT = ("Inter", 21)

class CosmeticsWizard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cosmetics Wizard")
        self.geometry("1200x800")

        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Creates the title
        self.header = ctk.CTkFrame(self, height=100, fg_color=BRIGHT_PURPLE, corner_radius=0)
        self.header.grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.header, text="COSMETICS WIZARD", font=FONT_HEADER, text_color="white")
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        # Creates the left panel with global worst and best ingredient information
        self.left_panel = ctk.CTkFrame(self, fg_color=BG_WHITE, corner_radius=15)
        self.left_panel.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.left_panel.grid_rowconfigure((0,1), weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.left_panel, text="Global Ingredients", font=FONT_SUBHEADER, text_color=TEXT_BLACK).pack(pady=(20,10))

        self.best_global = self.create_info_box(
            self.left_panel, "Top 5 Best",
            "• packette oryza sativa\n• cymbidium extract\n• jasminum sambac\n• amaranthus squalane\n• capric triglycerides",
            BRIGHT_PURPLE
        )
        self.worst_global = self.create_info_box(
            self.left_panel, "Top 5 Worst",
            "• calcium sulfate\n• sodium acrylate\n• pleiogynium fruit\n• podocarpus elatus\n• grevillea flower",
            BRIGHT_PURPLE
        )

        # Creates the center panel where the user can enter specific ingredients to analyze
        self.center_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.center_panel.grid(row=1, column=1, padx=10, pady=20, sticky="nsew")
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        self.input_label = ctk.CTkLabel(self.center_panel, text="Ingredient Analysis", font=FONT_LABEL)
        self.input_label.pack(pady=(10,5))

        self.input_box = ctk.CTkTextbox(self.center_panel, corner_radius=15, border_width=2, font=FONT_LABEL)
        self.input_box.pack(fill="both", expand=True, pady=10)

        self.submit_btn = ctk.CTkButton(self.center_panel, text="Analyze Formula", height=50,
                                        command=self.handle_prediction, font=FONT_BUTTON,
                                        fg_color=BRIGHT_PURPLE, hover_color=LIGHT_PURPLE)
        self.submit_btn.pack(fill="x", pady=10)

        # Creates the right panel that contains the product rating and best/worst ingredients
        self.right_panel = ctk.CTkFrame(self, fg_color=BG_WHITE, corner_radius=15)
        self.right_panel.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")
        self.right_panel.grid_rowconfigure((0,1,2), weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.res_rating_box = ctk.CTkFrame(self.right_panel, fg_color="white", corner_radius=10)
        self.res_rating_box.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.res_rating_box.grid_rowconfigure(0, weight=1)
        self.res_rating_box.grid_columnconfigure(0, weight=1)

        self.rating_val = ctk.CTkLabel(self.res_rating_box, text="-", font=FONT_RATING, text_color=BRIGHT_PURPLE)
        self.rating_val.place(relx=0.5, rely=0.4, anchor="center")

        ctk.CTkLabel(self.res_rating_box, text="Predicted Rating", text_color="gray").place(relx=0.5, rely=0.7, anchor="center")

        self.res_best = self.create_result_label(self.right_panel, "Best in this formula", "green")
        self.res_worst = self.create_result_label(self.right_panel, "Worst in this formula", "#CC0000")

    def create_info_box(self, parent, title, content, color):
        frame = ctk.CTkFrame(parent, fg_color=color, corner_radius=12)
        frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(frame, text=title, font=FONT_INFO_TITLE, text_color="white").pack(pady=5)
        ctk.CTkLabel(frame, text=content, justify="left", text_color="#E0E0E0", font=FONT_INFO_CONTENT, anchor="nw").pack(fill="both", expand=True, padx=10, pady=5)
        return frame

    def create_result_label(self, parent, title, color):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(sticky="nsew", padx=10, pady=5)

        ctk.CTkLabel(frame, text=title, font=FONT_RESULT_TITLE, text_color=TEXT_BLACK).grid(row=0, column=0, sticky="w")
        content = ctk.CTkLabel(frame, text="-", text_color=color, justify="left", wraplength=250, font=FONT_RESULT_CONTENT)
        content.grid(row=1, column=0, sticky="w")
        return content

    def handle_prediction(self):
        ingredients = self.input_box.get("1.0", "end-1c")
        if not ingredients.strip():
            return

        results = predict_and_explain(ingredients, label="Moisturizer", oily=1)
        self.rating_val.configure(text=results.get("rating"))
        self.res_best.configure(text="\n".join([f"• {i}" for i in results.get("best")]))
        self.res_worst.configure(text="\n".join([f"• {i}" for i in results.get("worst")]))

# Runs the program
if __name__ == "__main__":
    app = CosmeticsWizard()
    app.mainloop()
