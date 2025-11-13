# Human-Centric-Evolutionary-Art-through-Generative-Optimization-And-its-Explainability
# **Human-Centered Multi-Shape Evolutionary Art (PSO-Guided)**  
*A generative optimization framework where humans and algorithms co-create visual art through iterative evolution.*

---

## **🎨 Overview**

<div align="justify">

This project redefines evolutionary art by placing the **human user at the center of the creative loop**. Instead of letting an algorithm blindly optimize toward a target, the system blends **Particle Swarm Optimization (PSO)** with **direct human steering**, enabling artwork to evolve from simple geometric primitives into a fully reconstructed visual composition.

Each stroke—whether rectangle, circle, triangle, or line—is treated as an independently optimized entity. The PSO engine searches for the best version of that stroke (its position, angle, color, opacity, and size) while the human periodically influences the direction by choosing preferred shapes or pushing the system toward warmer colors, stronger contrast, lighter opacity, and more. The result is a **hybrid creative workflow** where computational precision meets human intuition.
</div>


---

## **🌀 How the Evolution Works**
The algorithm reconstructs the target image **shape by shape**, but not all at once—each shape is generated through its own small PSO process:

- **A particle = one possible shape**, defined by nine parameters  
  *(x, y, width, height, angle, R, G, B, opacity)*  
- **The canvas updates sequentially**: when the best particle is found, that shape is permanently drawn.  
- **Human feedback shapes the evolution**: after fixed intervals, the user chooses \*what direction to evolve next\*.  
- **Generative emergence**: early shapes capture coarse structure, while later shapes refine fine-grained detail.  
- **160 evolutionary steps** (by default) balance efficiency with visual richness.

This produces a visual “growth” effect: the image emerges layer by layer, echoing how traditional artists build form through successive strokes.

---

## **👤 Human-in-the-Loop Creativity**
Human feedback is not a side feature—it is the **driving force** behind the aesthetic direction.  
At interactive checkpoints, the user can:

- switch the preferred shape type,  
- adjust color tendencies (warmer/cooler/contrast),  
- influence opacity and stroke size.

This intentionally blends machine optimization with human taste, allowing each artwork to diverge from pure reconstruction and reflect **authorship, intention, and personality**. The system becomes a creative partner rather than a tool.

---

## **📈 Evolution Dynamics**
As the evolutionary cycle progresses:

- Early shapes approximate low-frequency structure (background, silhouettes).  
- Mid-phase shapes refine local regions and improve color distribution.  
- Final shapes correct micro-details through thin lines, small triangles, and low-opacity touches.  
- Error decreases rapidly at each shape, demonstrating PSO’s strong convergence for local image reconstruction.

The cumulative effect is an artwork that is both **algorithmically precise** and **artistically guided**.

---

## **🧠 Light Explainability (Optional)**
A minimal explainability layer is included to help users understand which parameters influenced reconstruction:

- **SHAP** gives a global ranking of parameter influence.  
- **Permutation Importance** validates which parameters mattered most overall.  
- **LIME** explains why one specific shape behaved the way it did.

Across all methods, **opacity consistently emerges as the most powerful artistic driver**, shaping how strokes blend into the evolving canvas.

Explainability in this project is not for debugging—it is to make creative decisions transparent and to show how algorithmic “understanding” aligns with human perception.

---

## **🖼 Output Files**
The project saves a complete record of the evolution, including:

- final canvas (`canvas_final.png`)  
- intermediate snapshots (visual evolution frames)  
- fitness trend over shapes  
- error contribution per shape  
- explainability visualizations (SHAP, LIME, Permutation)  
- combined importance CSV  

These files document both the *artistic process* and the *evolutionary logic* behind the creation.

---

## **🚀 Highlights**
- Fully interactive **human-guided generative art system**  
- Multi-shape evolutionary rendering with alpha blending  
- Fast PSO-based convergence for local stroke optimization  
- Strong integration of human aesthetic preference  
- Optional XAI support for transparent creative decision-making  
- Automatic generation of all evolution plots and logs  

---

## **📜 Citation**
**S. M. H. Mousavi — Human-Centered Evolutionary Art Through Generative Optimization and Its Explainability**

---

## **📧 Contact**
For collaboration or inquiries:  
**Seyed Muhammad Hossein Mousavi**  


