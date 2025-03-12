#set document(title: "Joe Lin")
#set page(margin: (0.5in))

#align(center)[
  #block(
    text(weight: 700, 1.5em, "Joe Lin"),
    below: 0.8em
  )
]

#set par(justify: true)

#show heading.where(level: 1): it => [
  #set text(13pt, weight: "bold")
  #block(
    width: 100%,
    stroke: (bottom: 0.5pt),
    inset: (bottom: 0.5em),
    above: 1em,
    below: 0.8em,
    smallcaps(it.body)
  )
]

#show heading.where(level: 2): set text(12pt, weight: "bold")

#let entry(l, r, below: 0.8em) = block(
  below: below,
  grid(
    columns: (1fr, auto),
    align(left, l),
    align(right, r),
  )
)

#let hstack(..items) = stack(
  dir: ltr,
  spacing: 0.5em,
  ..items.pos().map(x => align(bottom, x))
)

#let fab(name) = text(font: "Font Awesome 6 Brands", name)
#let fas(name) = text(font: "Font Awesome 6 Free Solid", name)
#let far(name) = text(font: "Font Awesome 6 Free", name)

#align(center)[
  #link("mailto:joelintech@ucla.edu")[`joelintech@ucla.edu`] #h(0.5em)•#h(0.5em) (408)-595-3211 #h(0.5em)•#h(0.5em) #link("https://www.linkedin.com/in/joe-lin-tech/")[#fab("linkedin")` joe-lin-tech`] #h(0.5em)•#h(0.5em) #link("https://github.com/joe-lin-tech")[#fab("github")` joe-lin-tech`]
]

= Education
#entry[
  == University of California Los Angeles \
  Computer Science, B.S.
][09/22-06/26 \ Major GPA: 4.0]
*Coursework*: Deep Learning, Computer Vision, Diffusion Models, Reinforcement Learning, Computer Organization, Autonomous Rover, Operating System Principles, Graphics, Linear Algebra, Statistics, Real Analysis, Differential Equations \
*Societies*: Upsilon Pi Epsilon (CS Honor Society), Tau Beta Pi (Engineering Honor Society)

= Experience
#entry[
  #hstack[== UCLA Learning Assistant Program][|][_New Learning Assistant_]
][09/24-12/24]
- Reflected on strategies for collaborative learning and sharpened skills in fostering peer engagement and discussion #footnote[In my LA feedback sheet, all responses selected _Agree_ or _Strongly Agree_ for questions related to engaging with all students in the classroom as well as fostering collaboration and discussion (e.g. "My LA helps to create an environment in my group where every student engages.", "My LA supports me if I am struggling and frustrated.")] <1>
- Prepared midway and final review slides to guide peer learning and foster collaborative discussion #footnote[For reference, here are the #link("https://drive.google.com/file/d/1EzfqfL9NGXy8W6XCYA0RPbi4x4vTxvjx/view?usp=sharing")[#text(blue)[midway]] and #link("https://drive.google.com/file/d/1JB30TT2b4LvDbyVZKHZqAQE1g971M1Y5/view?usp=sharing")[#text(blue)[final]] review slides. They serve as a method for checking peer understanding in a collaborative setting. With every question or walkthrough in the review slides, I first ask peers to work together with their groups, then ask for ideas to approach the problem, and essentially have peers work together as an entire discussion to discover the solution together. The results of this strategy are reflected by peers responding with _Agree_ or _Strongly Agree_ for "My LA ASKS me why something is true more often than they EXPLAIN to me why." Part of this is also evidenced by peer LAs selecting _Most of the time_ for "Redirects questions to foster collaboration", _Agree_ for "Asks a mix of open and closed questions to encourage engagement.", and _Always_ for "Asks closed questions to check for understanding before moving on." Whenever possible, I also check in with peers individually and ask about their progress on assignments and lecture content comprehension. All of these strategies combined proved effective as all peer responses marked _Agree_ or _Strongly Agree_ for "An LA checks in on my understanding in every section, especially if I am struggling." on my LA feedback form.] <2>
- Posed probing or redirecting questions to guide peers to uncover solutions to problems on their own @2
- Refined strategies for peer engagement to establish a more inclusive learning environment #footnote[From one response on the feedback form, it was suggested that I should avoid cold-calling people, even if its just to ask for ideas. To revise my strategy for engaging with quieter students, I learned to refocus on directly interacting/checking in with peers so they don't feel pressured to share with the entire group and feel more comfortable with their own ideas. After taking this step back, I observed a few of the quieter students speaking up voluntarily when questions are posed towards the end of quarter, which was super encouraging!] <3>
- Gained confidence in learning course material through discovering new perspectives with peers #footnote[All responses from the LA feedback form selected _Agree_ or _Strongly Agree_ for "My LA is comfortable/familiar with the course material (and/or asks TA when needed)." In addition, upon conversing with peers in the discussion, I learned a lot about where peers struggled and worked together with them to overcome these challenges, which, in a way, helped me gain a deeper understanding of course concepts and uncover new perspectives as well (e.g. view of transformer architectures).] <4>
- Recognized diversity of peer backgrounds through collaborative seminars and worked towards establishing an inclusive learning environment #footnote[In my LA feedback responses, all responses selected _Strongly Agree_ for "My LA has helped me feel more like I belong in STEM." As mentioned, this new awareness was also derived from the discussions we had during the seminars and assignments completed throughout relevant weeks of the LA program.] <5>
- Suggested increased opportunities for LAs to be active in peer's learning during discussion sessions #footnote[This is mainly a general thought that came up while reflecting on Part C. Because of the current nature of the coruse, it's difficult for LAs to be involved every discussion as the initial discussions are used by TAs to do a review on core software tools and discussions near the end are for course projects. In the future, I suggested that LAs can be more involved in every week's discussion (e.g. collaborative lecture review) and even making assignment guides or worksheets. I believe the LA program serves a critical role in supporting peer learning, but more students can become even more tightly involved with the course curriculum to further bolster the learning environment (as done at other schools).] <6>
- Enhanced communication skills through leading discussions and consistently guiding peers through their thinking #footnote[Overall, leading the review sessions gave me a ton of learning opportunities with respect to how to communicate new ideas to peers and checking for understanding. In several instances during discussion and Piazza, I had peers explain their current thinking step-by-step and through this process we were able to pinpoint bugs relatively quickly and through the peer's own efforts. This process involved a lot of learning in regards to communicating the right questions that effectively guide the peer's thinking in the right direction. A specific evidence of my growth in communication comes from the LA feedback form where one peer stated "Good speaker and actually engages with audience."] <7>
- Pursued meaningful and constructive interactions with instructors and peers #footnote[I've become more comfortable interacting with the instructors and peers throughout the quarter whether thats to make suggestions on the current format, pitch ideas, or checking in on peers for any help. In terms of peers, all of my LA feedback form responses selected _Agree_ or _Strongly Agree_ for "My LA is helpful to my learning in this course.", "My LA supports me if I am struggling and frustrated.", and "My LA uses my name when interacting with me." felt that each of them felt supported on an individual level.] <8>
- Progressed on SMARTER goal of encouraging more peer collaborations by arranging at least 5 group discussion times during the final review session for peers to discuss and work together #footnote[With the structuring of my final review slides, I was able to have students discuss questions together 6 times with around 2-5 minutes each (depending on the type of problem). I observed peers actively collaborating and discussing ideas with each other.]

= Additional Information
*Skills*: communication @7, fostering peer collaboration @2, establishing inclusive learning environments @5, listening \
*Interests*: recognizing diversity in student backgrounds @5, strategies for peer engagement and collaboration @2
